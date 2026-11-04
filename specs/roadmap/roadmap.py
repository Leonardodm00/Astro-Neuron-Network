"""Roadmap of a subproject: theory stages and implementation stages as two
connected graphs, with the implementation colours computed from the
repository.

One YAML file per subproject (specs/roadmap/<name>.yaml) is the source of
truth: the stages, their dependencies, and the theory statuses (set by
hand). The implementation statuses are never written by hand: this script
derives them from git, specs/SPEC.md, the integration-tester reports and
the HPC job markers on the hpc-status branch.

Usage
    python roadmap.py init   --name N --branch B [--repo OWNER/NAME]
                             [--plan PLAN.yaml] [--spec specs/SPEC.md] [--root PATH]
    python roadmap.py add    --yaml specs/roadmap/N.yaml block  ID --block N
                             --title T --parent STEP [--summary S]
                             [--depends-on I1,I2] [--needs-theory T1] [--needs-hpc]
    python roadmap.py add    --yaml specs/roadmap/N.yaml topic  ID --title T
                             --parent TSTEP [--summary S] [--refs R1,R2]
                             [--depends-on T1] [--status open|studying|closed]
    python roadmap.py add    --yaml specs/roadmap/N.yaml step   ID --title T
                             [--summary S] [--theory S2] [--depends-on S1] [--needs-hpc]
    python roadmap.py revise --yaml specs/roadmap/N.yaml STAGE --from OTHER
                             [--decision D-nnn] [--note TEXT]   |  STAGE --clear
    python roadmap.py check  --yaml specs/roadmap/N.yaml [--root PATH]
    python roadmap.py status --yaml specs/roadmap/N.yaml [--root PATH]
                             [--json OUT] [--no-fetch] [--source LABEL]
    python roadmap.py render --yaml specs/roadmap/N.yaml [--root PATH]
                             [--status STATUS.json] [--out specs/roadmap/N.md]
    python roadmap.py set    --yaml specs/roadmap/N.yaml STAGE STATUS
                             [--note TEXT] [--decision D-nnn] [--on DATE]

    --root is the repository root (default: the root of the repository that
    holds the YAML file, else the current directory).

Implementation states (one colour each; the page and the Mermaid rendering
take them from STATES below):
    planned   red      no code: the SPEC entry is drafted, or the modules
                       the entry names do not exist on the branch
    local     orange   code exists but is not on origin (uncommitted changes
                       or commits not pushed); only visible from a working
                       clone
    pushed    amber    on origin, no tester report covers the current code
    testing   blue     the night tester is running with this block in scope
    tested    teal     tester finished and found no BLOCKER/MAJOR; the HPC
                       run is still missing (stages with needs_hpc)
    running   violet   an HPC job of this stage is running (marker on the
                       hpc-status branch)
    failing   crimson  an open BLOCKER/MAJOR finding on the current code, or
                       the last HPC job failed, or its running marker is
                       stale
    done      green    pushed, tested, and (if needs_hpc) the HPC run
                       finished on the current code

Theory states: open (grey), studying (blue), closed (green).

Theory and implementation progress independently: a topic can be closed
before its block exists, a block can be done while its topic is still open.
The two sides are linked, not conditioned on each other: an implementation
stage names the topics it rests on (needs_theory, inherited from its step),
a topic names the stages its outcome shapes (affects); both appear as
theory_links / theory_open on the stage and as dotted edges. blocked_by
(dashed border) is reported only within a side: an implementation stage
whose dependency's code is not in place (planned, local, failing), a topic
whose prerequisite topic is not closed.

Revisions carry a decision across the sides: `revise I4 --from T7
--decision D-082 --note ...` flags I4 as having to change because of T7 (the
flag is drawn on the node and listed in the detail); it clears by itself
when I4's code changes after the flag's commit, or with `revise I4 --clear`.
A topic can be flagged from a block the same way; its flag clears when the
topic's status changes.

Groups: a stage with `parent: <id>` is a member of that stage, to any
depth. A stage with members is drawn as a frame around them, coloured by
its least advanced member (failing < planned < local < pushed < testing <
tested < running < done; open < studying < closed), with its own node
inside when it has a block of its own. An implementation stage without a
block is a pure group: it has no state of its own and is not counted.

Plan first: a project starts as a coarse list of steps, and the blocks
of a step are defined while the step is carried out. `init --plan
PLAN.yaml` writes each step S<k> as an implementation group with no blocks
yet (drawn planned, red, until its first block arrives). A theory group
TS<k> is created only for a step whose plan entry names the concepts it
rests on (`theory:`, mathematics, physics, biology, engineering); routines
(I/O, configuration, production runs, acceptance tests) have no theory
counterpart, and several blocks may rest on one theory stage. `add block`
/ `add topic` then hang a SPEC block or a topic under its group as the
work proceeds. PLAN.yaml:

    steps:
      - id: S1
        title: Forward model
        summary: what the step delivers
        theory: the mathematics or physics underlying it   # omit for a routine
        depends_on: [S0]                                   # other steps
        needs_hpc: false
        blocks: [0, 1]                                     # SPEC blocks already written, if any

Exit codes: 0 on success, 1 on a validation error, 2 if git fails.
"""
from __future__ import annotations

import argparse
import datetime as dt
import fnmatch
import json
import os
import re
import shutil
import subprocess
import sys

try:
    import yaml
except ImportError:  # pragma: no cover
    yaml = None

SCHEMA_VERSION = 1

# ----------------------------------------------------------------- states ---

# (state, fill colour, text colour, meaning)
STATES = {
    "implementation": [
        ("planned", "#d62828", "#ffffff", "not implemented"),
        ("local", "#f77f00", "#1a1a1a", "implemented, not pushed"),
        ("pushed", "#e9c46a", "#1a1a1a", "pushed, not tested"),
        ("testing", "#3a86ff", "#ffffff", "independent tester running"),
        ("tested", "#2a9d8f", "#ffffff", "tester passed, HPC run missing"),
        ("running", "#8338ec", "#ffffff", "HPC job running"),
        ("failing", "#b5174e", "#ffffff", "tester or HPC failure on the current code"),
        ("done", "#2d9a3d", "#ffffff", "pushed, tested, HPC run done"),
    ],
    "theory": [
        ("open", "#8d99ae", "#1a1a1a", "not started"),
        ("studying", "#3a86ff", "#ffffff", "being studied"),
        ("closed", "#2d9a3d", "#ffffff", "closed"),
    ],
}
THEORY_STATES = [s[0] for s in STATES["theory"]]
IMPL_STATES = [s[0] for s in STATES["implementation"]]

BLOCK_HEADER = re.compile(r"^###\s+Block\s+(\d+)\s*:\s*(.*?)\s*$")
FIELD_LINE = re.compile(r"^-\s+([A-Za-z][A-Za-z /]*?)\s*:")
BACKTICKED = re.compile(r"`([^`]+)`")
BLOCK_REFS = re.compile(r"\bBlocks?\s+(\d+(?:\s*(?:,|and|&|-)\s*\d+)*)")
REPORT_COMMIT = re.compile(r"^Commit:\s*([0-9a-fA-F]{7,40})\b", re.MULTILINE)
REPORT_MODE = re.compile(r"^Mode:\s*(\w+)", re.MULTILINE)
REPORT_SCOPE = re.compile(r"^Scope:\s*(.*)$", re.MULTILINE)
REPORT_DATE = re.compile(r"^#.*?(\d{4}-\d{2}-\d{2})", re.MULTILINE)
VERDICT = re.compile(r"^##\s+Verdict\s*\n\s*(PASS WITH ISSUES|PASS|FAIL)",
                     re.MULTILINE)
STATUS_FIELD = re.compile(r"Run status:\s*(\w+)")
STATUS_COMMIT = re.compile(r"Commit under test:\s*([0-9a-fA-F]{7,40})")
STATUS_MODE = re.compile(r"Mode:\s*(\w+)")
STATUS_STARTED = re.compile(r"Started \(UTC\):\s*([0-9: -]+)")
RESOLVED_WORDS = re.compile(
    r"not a problem|resolved|fixed|closed|withdrawn|superseded|no longer",
    re.IGNORECASE)
HPC_WORDS = re.compile(r"davinci|cluster|\bHPC\b|\bPBS\b|qsub|sbatch|SLURM",
                       re.IGNORECASE)


class RoadmapError(RuntimeError):
    pass


class GitError(RoadmapError):
    pass


# -------------------------------------------------------------------- git ---

def git(root, *args, check=True):
    result = subprocess.run(["git", "-C", root, *args],
                            capture_output=True, text=True)
    if check and result.returncode != 0:
        raise GitError("git %s: %s" % (" ".join(args), result.stderr.strip()))
    return result.stdout


def git_ok(root, *args):
    return subprocess.run(["git", "-C", root, *args],
                          capture_output=True, text=True).returncode == 0


def resolve_rev(root, rev):
    out = subprocess.run(["git", "-C", root, "rev-parse", "--verify",
                          "--quiet", rev + "^{commit}"],
                         capture_output=True, text=True)
    return out.stdout.strip() if out.returncode == 0 else None


def read_at(root, rev, path):
    out = subprocess.run(["git", "-C", root, "show", "%s:%s" % (rev, path)],
                         capture_output=True, text=True)
    return out.stdout if out.returncode == 0 else None


def is_ancestor(root, older, newer):
    return git_ok(root, "merge-base", "--is-ancestor", older, newer)


def tree_paths(root, rev):
    return git(root, "ls-tree", "-r", "--name-only", rev).splitlines()


def repo_root_of(path):
    out = subprocess.run(["git", "-C", os.path.dirname(os.path.abspath(path))
                          or ".", "rev-parse", "--show-toplevel"],
                         capture_output=True, text=True)
    return out.stdout.strip() if out.returncode == 0 else None


def now_iso():
    return dt.datetime.now(dt.timezone.utc).replace(microsecond=0).isoformat()


# ------------------------------------------------------------------- yaml ---

def need_yaml():
    if yaml is None:
        raise RoadmapError("PyYAML is required: pip install pyyaml")


def load_roadmap(path):
    need_yaml()
    with open(path, "r", encoding="utf-8") as fh:
        data = yaml.safe_load(fh) or {}
    data.setdefault("theory", [])
    data.setdefault("implementation", [])
    for stage in data["theory"]:
        # a theory stage with members and no status of its own is a pure
        # group: its colour is its least advanced member's
        stage["status_given"] = bool(stage.get("status"))
        stage["status"] = stage.get("status") or "open"
        stage["depends_on"] = as_list(stage.get("depends_on"))
        stage["refs"] = as_list(stage.get("refs"))
        stage["summary"] = stage.get("summary") or ""
        stage["parent"] = stage.get("parent") or None
        stage["affects"] = as_list(stage.get("affects"))
    data["revisions"] = [r for r in (data.get("revisions") or []) if isinstance(r, dict)]
    for stage in data["implementation"]:
        stage["depends_on"] = as_list(stage.get("depends_on"))
        stage["needs_theory"] = as_list(stage.get("needs_theory"))
        stage["needs_hpc"] = bool(stage.get("needs_hpc"))
        stage["jobs"] = as_list(stage.get("jobs"))
        stage["summary"] = stage.get("summary") or ""
        stage["parent"] = stage.get("parent") or None
        blocks = stage.get("blocks")
        if blocks is None:
            block = stage.get("block")
            blocks = [] if block is None else [block]
        stage["blocks"] = [int(b) for b in blocks]
    data.setdefault("spec", "specs/SPEC.md")
    data.setdefault("hpc_branch", "hpc-status")
    data.setdefault("stale_after_hours", 48)
    return data


def as_list(value):
    if value is None:
        return []
    if isinstance(value, (list, tuple)):
        return list(value)
    return [value]


def check_roadmap(data):
    """Return a list of problems (empty when the roadmap is consistent)."""
    problems = []
    for key in ("name", "branch"):
        if not data.get(key):
            problems.append("missing top-level field: %s" % key)
    theory = {s.get("id"): s for s in data["theory"]}
    impl = {s.get("id"): s for s in data["implementation"]}
    for kind, stages in (("theory", data["theory"]),
                         ("implementation", data["implementation"])):
        seen = set()
        for stage in stages:
            sid = stage.get("id")
            if not sid or not re.match(r"^[A-Za-z][A-Za-z0-9_.-]*$", str(sid)):
                problems.append("%s stage without a valid id: %r" % (kind, stage))
                continue
            if sid in seen:
                problems.append("duplicate id %s" % sid)
            seen.add(sid)
            if not stage.get("title"):
                problems.append("%s has no title" % sid)
            if kind == "theory" and stage["status"] not in THEORY_STATES:
                problems.append("%s: status %r not in %s"
                                % (sid, stage["status"], THEORY_STATES))
            if kind == "theory":
                for ref in as_list(stage.get("affects")):
                    if ref not in impl:
                        problems.append("%s affects %s: no such implementation stage" % (sid, ref))
            if kind == "implementation":
                manual = stage.get("manual_status")
                if manual is not None and manual not in IMPL_STATES:
                    problems.append("%s: manual_status %r not in %s"
                                    % (sid, manual, IMPL_STATES))
                for ref in as_list(stage.get("needs_theory")):
                    if ref not in theory:
                        problems.append("%s needs_theory %s: no such theory stage"
                                        % (sid, ref))
            pool = theory if kind == "theory" else impl
            for ref in as_list(stage.get("depends_on")):
                if ref not in pool:
                    problems.append("%s depends_on %s: no such %s stage"
                                    % (sid, ref, kind))
                elif ref == sid:
                    problems.append("%s depends on itself" % sid)
            parent = stage.get("parent")
            if parent is not None:
                if parent not in pool:
                    problems.append("%s parent %s: no such %s stage" % (sid, parent, kind))
                elif parent == sid:
                    problems.append("%s is its own parent" % sid)
    if set(theory) & set(impl):
        problems.append("ids shared between theory and implementation: %s"
                        % sorted(set(theory) & set(impl)))
    for rev in data.get("revisions", []):
        if rev.get("stage") not in theory and rev.get("stage") not in impl:
            problems.append("revision of unknown stage %r" % rev.get("stage"))
    for kind, pool in (("theory", theory), ("implementation", impl)):
        cycle = find_cycle({k: as_list(v.get("depends_on")) for k, v in pool.items()})
        if cycle:
            problems.append("%s dependencies form a cycle: %s" % (kind, " -> ".join(cycle)))
        cycle = find_cycle({k: ([v["parent"]] if v.get("parent") else [])
                            for k, v in pool.items()})
        if cycle:
            problems.append("%s parents form a cycle: %s" % (kind, " -> ".join(cycle)))
        for sid, stage in pool.items():
            parent = stage.get("parent")
            if parent in pool and parent in as_list(stage.get("depends_on")):
                problems.append("%s depends on its own group %s" % (sid, parent))
    return problems


def find_cycle(graph):
    state, stack = {}, []

    def visit(node):
        state[node] = 1
        stack.append(node)
        for nxt in graph.get(node, []):
            if nxt not in graph:
                continue
            if state.get(nxt) == 1:
                return stack[stack.index(nxt):] + [nxt]
            if nxt not in state:
                found = visit(nxt)
                if found:
                    return found
        stack.pop()
        state[node] = 2
        return None

    for node in graph:
        if node not in state:
            found = visit(node)
            if found:
                return found
    return None


# ------------------------------------------------------------------- spec ---

def parse_spec(text):
    """{number: {"name": str, "lines": [...], "text": str}}."""
    blocks, current, field = {}, None, None
    for line in (text or "").splitlines():
        header = BLOCK_HEADER.match(line)
        if header:
            current = {"name": header.group(2), "lines": [], "raw": []}
            blocks[int(header.group(1))] = current
            field = None
            continue
        if line.startswith("#"):
            current, field = None, None
            continue
        if current is None:
            continue
        match = FIELD_LINE.match(line)
        if match:
            field = match.group(1).strip().lower()
        elif not line.strip() or not line.startswith((" ", "\t")):
            field = None
        current["lines"].append((field, line.rstrip()))
        current["raw"].append(line.rstrip())
    for block in blocks.values():
        block["text"] = "\n".join(block["raw"])
    return blocks


def field_text(block, *names):
    return " ".join(line.strip() for field, line in block["lines"]
                    if field in names)


SPEC_STATUS_WORDS = ("integration-tested", "smoke-tested", "implemented", "drafted",
                     "verified")
# longest alternatives first, so "smoke-tested" is one match, not "tested";
# "not implemented" does not count as implemented
SPEC_STATUS_RE = re.compile(r"(?<!not )(?<!un)\b(%s)\b" % "|".join(SPEC_STATUS_WORDS),
                            re.IGNORECASE)


def spec_status_word(block):
    """The block's current status word from its Status line.

    A Status line is a history ("drafted 2026-10-21; 14.1-14.6 implemented
    2026-10-21; gate 2 next"), so the LAST known status word in it is the
    current one. An entry can also hold other "Status:" lines (a fit's
    status field, say); words outside the known set are ignored when a
    known one exists.
    """
    text = field_text(block, "status")
    if not text:
        # no "- Status:" field: the status is embedded in another line
        # ("... and the runbook. Status: drafted 2026-10-21 ..."); take the
        # text after the last "Status:" of the entry
        holders = [line for line in block["raw"] if "Status:" in line]
        text = holders[-1].split("Status:", 1)[1] if holders else ""
    known = [m.group(1).lower() for m in SPEC_STATUS_RE.finditer(text)]
    if known:
        return known[-1]
    first = re.findall(r"Status:\s*(\w[\w-]*)", text)
    return first[0].lower() if first else ""


def looks_like_path(token):
    token = token.strip()
    if ":" in token or "(" in token or "=" in token or " " in token:
        return False
    return "/" in token or bool(re.search(r"\.[A-Za-z0-9]+$", token))


def module_tokens(block):
    """Backticked paths of the Module/Modules field that are the block's own.

    Modules are listed comma-separated; parenthesised annotations ("(the
    fitter)", "(`measure_node`)") and everything after the first semicolon
    outside parentheses ("; selected by ... through `scripts/end_to_end.py`")
    are prose about consumers, not modules of the block.
    """
    text = field_text(block, "module", "modules")
    previous = None
    while previous != text:
        previous = text
        text = re.sub(r"\([^()]*\)", "", text)
    text = text.split(";")[0]
    return [t.strip() for t in BACKTICKED.findall(text) if looks_like_path(t)]


def coverage_paths(text):
    paths, inside, found = [], False, False
    for line in (text or "").splitlines():
        if line.startswith("#"):
            inside = bool(re.match(r"^#{1,6}\s+.*\bCoverage\b", line, re.I))
            found = found or inside
            continue
        if inside:
            for snippet in BACKTICKED.findall(line):
                token = snippet.strip()
                if token and not token.startswith("<") and looks_like_path(token):
                    paths.append(token)
    return paths if found else []


def resolve_modules(tokens, paths, coverage):
    """Map the backticked module tokens of a SPEC entry to paths of the tree.

    A token is taken as is when the tree has it; otherwise as a suffix
    (`analysis/fit.py` -> `<...>/src/allen_diameter/analysis/fit.py`),
    preferring paths under the coverage and then the shortest path. A token
    ending in "/" claims a directory; a token with * ? [ is a glob.
    """
    resolved, unresolved = [], []
    path_set = set(paths)
    for token in tokens:
        if token in path_set:
            resolved.append(token)
            continue
        if token.endswith("/"):
            hits = [p for p in paths if p.startswith(token)
                    or ("/" + token) in p]
            dirs = sorted({p[:p.index(token) + len(token)] for p in hits
                           if token in p})
            if dirs:
                resolved.extend(dirs)
                continue
        if any(ch in token for ch in "*?["):
            hits = [p for p in paths if fnmatch.fnmatchcase(p, token)
                    or fnmatch.fnmatchcase(os.path.basename(p), token)]
            if hits:
                resolved.extend(hits)
                continue
        hits = [p for p in paths if p.endswith("/" + token)]
        if hits:
            covered = [p for p in hits if any(claims(c, p) for c in coverage)]
            pool = covered or hits
            pool.sort(key=lambda p: (len(p), p))
            resolved.append(pool[0])
            continue
        unresolved.append(token)
    return sorted(set(resolved)), unresolved


def claims(prefix, path):
    prefix = prefix.strip()
    if any(ch in prefix for ch in "*?["):
        return fnmatch.fnmatchcase(path, prefix)
    if prefix.endswith("/"):
        return path.startswith(prefix)
    return path == prefix or path.startswith(prefix + "/")


def block_refs(text):
    """Block numbers named as "Block 3", "Blocks 2, 4 and 7", "Blocks 0-2".

    Parenthesised annotations are dropped first, so "Blocks 0 (real
    provider), 3 (fit), 4 (synthetic provider)" gives {0, 3, 4}.
    """
    numbers = set()
    text = re.sub(r"\([^)]*\)", "", text or "")
    for match in BLOCK_REFS.finditer(text):
        chunk = match.group(1)
        ranges = re.findall(r"(\d+)\s*-\s*(\d+)", chunk)
        for lo, hi in ranges:
            numbers.update(range(int(lo), int(hi) + 1))
        chunk = re.sub(r"\d+\s*-\s*\d+", "", chunk)
        numbers.update(int(n) for n in re.findall(r"\d+", chunk))
    return numbers


def consumed_blocks(block):
    """Block numbers named after 'consumes' on the Data flow line."""
    text = field_text(block, "data flow", "consumes / produces for")
    match = re.search(r"consumes(.*?)(?:;|feeds|$)", text, re.I | re.S)
    return block_refs(match.group(1)) if match else set()


# ---------------------------------------------------------- tester reports ---

def parse_report(text, path):
    commit = REPORT_COMMIT.search(text or "")
    if not commit:
        return None
    mode = REPORT_MODE.search(text)
    scope = REPORT_SCOPE.search(text)
    date = REPORT_DATE.search(text)
    verdict = VERDICT.search(text)
    report = {
        "path": path,
        "commit": commit.group(1),
        "mode": (mode.group(1).lower() if mode else "full"),
        "scope": (scope.group(1).strip() if scope else ""),
        "date": (date.group(1) if date else ""),
        "verdict": (verdict.group(1) if verdict else ""),
        "findings": [],
        "decisions": {},
    }
    scope_text = report["scope"]
    report["scope_blocks"] = (None if report["mode"] == "full"
                              or "whole" in scope_text.lower()
                              else block_refs(scope_text))
    report["findings"] = parse_findings(text)
    report["decisions"] = parse_decisions(text)
    return report


def table_rows(text, heading_regex):
    """Rows (list of cells) of the first table under a heading."""
    rows, inside, header_seen = [], False, False
    for line in (text or "").splitlines():
        if line.startswith("#"):
            inside = bool(re.match(heading_regex, line, re.I))
            header_seen = False
            continue
        if not inside:
            continue
        if line.strip().startswith("|"):
            cells = [c.strip() for c in line.strip().strip("|").split("|")]
            if not header_seen:
                header_seen = True  # header row
                continue
            if all(re.match(r"^:?-+:?$", c) for c in cells if c):
                continue
            rows.append(cells)
        elif rows:
            break
    return rows


def parse_findings(text):
    findings = []
    for cells in table_rows(text, r"^#{1,6}\s+Findings"):
        if len(cells) < 3 or not re.match(r"^F\d+", cells[0]):
            continue
        # "4, 6, 11", "0 -> 5 boundary", "1 (10)", "7 (gate 2 tool)":
        # drop parentheses that hold words, keep bare numbers
        cell = re.sub(r"\([^)]*[A-Za-z][^)]*\)", "", cells[2])
        findings.append({
            "id": cells[0],
            "severity": cells[1].upper(),
            "blocks": sorted({int(n) for n in re.findall(r"\d+", cell)}),
            "finding": cells[4] if len(cells) > 4 else "",
        })
    return findings


def parse_decisions(text):
    decisions = {}
    for cells in table_rows(text, r"^#{1,6}\s+Triage"):
        if cells and re.match(r"^[FS]\d+", cells[0]) and len(cells) >= 2:
            decisions[cells[0]] = cells[-1]
    for cells in table_rows(text, r"^#{1,6}\s+Tester findings"):
        if cells and re.match(r"^[FS]\d+", cells[0]) and len(cells) >= 2:
            decisions.setdefault(cells[0], cells[-1])
    return decisions


def collect_reports(root, head, reports_dir="specs/test_reports"):
    """Reports on the branch head and on tester/* branches, newest first."""
    refs = [head] + git(root, "for-each-ref", "--format=%(refname)",
                        "refs/heads/tester", "refs/remotes/origin/tester").split()
    reports, decisions, seen = [], {}, set()
    for ref in refs:
        listing = subprocess.run(
            ["git", "-C", root, "ls-tree", "--name-only", ref, reports_dir + "/"],
            capture_output=True, text=True).stdout.splitlines()
        for path in listing:
            if not path.endswith(".md"):
                continue
            text = read_at(root, ref, path) or ""
            base = os.path.basename(path)
            if "triage" in base.lower():
                # a triage file (<date>_triage.md) belongs to the report(s)
                # of the same date; its table may sit under any heading
                day = base[:10]
                found = decisions.setdefault(day, {})
                for cells in table_rows_any(text):
                    if cells and re.match(r"^[FS]\d+", cells[0]) and cells[-1].strip():
                        found.setdefault(cells[0], cells[-1])
                continue
            report = parse_report(text, path)
            if report and report["commit"] not in seen:
                seen.add(report["commit"])
                reports.append(report)
    for report in reports:
        day = os.path.basename(report["path"])[:10]
        for fid, cell in decisions.get(day, {}).items():
            if not report["decisions"].get(fid, "").strip():
                report["decisions"][fid] = cell
    reports.sort(key=lambda r: (r["date"], r["path"]), reverse=True)
    return reports


def table_rows_any(text):
    rows, header_seen = [], False
    for line in (text or "").splitlines():
        if line.strip().startswith("|"):
            cells = [c.strip() for c in line.strip().strip("|").split("|")]
            if all(re.match(r"^:?-+:?$", c) for c in cells if c):
                continue
            if not header_seen:
                header_seen = True
                continue
            rows.append(cells)
        else:
            header_seen = False
    return rows


def tester_runs(root, reports_dir="specs/test_reports"):
    """Status files of tester/* branches: running runs and their commits."""
    runs = []
    refs = git(root, "for-each-ref", "--format=%(refname)",
               "refs/heads/tester", "refs/remotes/origin/tester").split()
    for ref in refs:
        listing = subprocess.run(
            ["git", "-C", root, "ls-tree", "--name-only", ref, reports_dir + "/"],
            capture_output=True, text=True).stdout.splitlines()
        for path in listing:
            if not path.endswith(".status"):
                continue
            text = read_at(root, ref, path) or ""
            status = STATUS_FIELD.search(text)
            commit = STATUS_COMMIT.search(text)
            if not status or not commit:
                continue
            mode = STATUS_MODE.search(text)
            started = STATUS_STARTED.search(text)
            runs.append({
                "branch": ref.replace("refs/remotes/origin/", "").replace("refs/heads/", ""),
                "status": status.group(1).lower(),
                "commit": commit.group(1),
                "mode": (mode.group(1).lower() if mode else "full"),
                "started": (started.group(1).strip() if started else ""),
            })
    return runs


# ------------------------------------------------------------ hpc markers ---

def hpc_markers(root, hpc_branch):
    """Markers runs/<stage>/<job>.json on the hpc-status branch (origin first)."""
    ref = None
    for candidate in ("refs/remotes/origin/" + hpc_branch, "refs/heads/" + hpc_branch):
        if resolve_rev(root, candidate):
            ref = candidate
            break
    if ref is None:
        return []
    markers = []
    for path in subprocess.run(
            ["git", "-C", root, "ls-tree", "-r", "--name-only", ref, "runs/"],
            capture_output=True, text=True).stdout.splitlines():
        if not path.endswith(".json"):
            continue
        try:
            marker = json.loads(read_at(root, ref, path) or "{}")
        except ValueError:
            continue
        marker["path"] = path
        marker.setdefault("stage", path.split("/")[1] if path.count("/") >= 2 else "")
        markers.append(marker)
    return markers


def parse_time(value):
    if not value:
        return None
    value = value.strip().replace("Z", "+00:00")
    try:
        parsed = dt.datetime.fromisoformat(value)
    except ValueError:
        try:
            parsed = dt.datetime.strptime(value, "%Y-%m-%d %H:%M")
        except ValueError:
            return None
    if parsed.tzinfo is None:
        parsed = parsed.replace(tzinfo=dt.timezone.utc)
    return parsed


# ------------------------------------------------------------------ status ---

def compute_status(data, root, fetch=True, source="chat"):
    branch = data["branch"]
    if fetch:
        subprocess.run(["git", "-C", root, "fetch", "--quiet", "origin",
                        "+refs/heads/tester/*:refs/remotes/origin/tester/*"],
                       capture_output=True, text=True)
        subprocess.run(["git", "-C", root, "fetch", "--quiet", "origin",
                        "+refs/heads/%s:refs/remotes/origin/%s"
                        % (data["hpc_branch"], data["hpc_branch"])],
                       capture_output=True, text=True)
        subprocess.run(["git", "-C", root, "fetch", "--quiet", "origin",
                        "+refs/heads/%s:refs/remotes/origin/%s" % (branch, branch)],
                       capture_output=True, text=True)

    head = resolve_rev(root, branch) or resolve_rev(root, "origin/" + branch)
    if head is None:
        raise GitError("branch %s not found locally or on origin" % branch)
    remote_head = resolve_rev(root, "origin/" + branch)
    current = git(root, "rev-parse", "--abbrev-ref", "HEAD").strip()
    worktree_is_branch = current == branch
    shallow = git(root, "rev-parse", "--is-shallow-repository").strip() == "true"

    spec_text = read_at(root, head, data["spec"])
    blocks = parse_spec(spec_text)
    coverage = coverage_paths(spec_text)
    paths = tree_paths(root, head)
    reports = collect_reports(root, head)
    runs = tester_runs(root)
    markers = hpc_markers(root, data["hpc_branch"])
    dirty = set()
    if worktree_is_branch:
        for line in git(root, "status", "--porcelain", "--untracked-files=no").splitlines():
            dirty.add(line[3:].split(" -> ")[-1].strip('"'))
    now = dt.datetime.now(dt.timezone.utc)
    stale_hours = float(data.get("stale_after_hours", 48))

    last_finished = max((r for r in reports), key=lambda r: r["date"], default=None)

    theory = {}
    for stage in data["theory"]:
        theory[stage["id"]] = {
            "id": stage["id"],
            "title": stage.get("title", stage["id"]),
            "summary": stage.get("summary", ""),
            "refs": as_list(stage.get("refs")),
            "depends_on": as_list(stage.get("depends_on")),
            "parent": stage.get("parent"),
            "status": stage.get("status", "open"),
            "closed_on": str(stage.get("closed_on") or ""),
            "decision": str(stage.get("decision") or ""),
            "note": stage.get("note", ""),
            "affects": as_list(stage.get("affects")),
            "has_own": stage.get("status_given", True),
        }
    # a theory stage without members always has a status of its own
    members_of = {}
    for stage in data["theory"]:
        if stage.get("parent") in theory:
            members_of.setdefault(stage["parent"], []).append(stage["id"])
    for sid, stage in theory.items():
        if sid not in members_of:
            stage["has_own"] = True

    impl = {}
    for stage in data["implementation"]:
        impl[stage["id"]] = evaluate_stage(
            stage, data, root, head, remote_head, worktree_is_branch, dirty,
            blocks, coverage, paths, reports, runs, markers, now, stale_hours,
            last_finished)

    # groups: children, depth, and the aggregate of the descendants' states
    attach_groups(theory, "theory")
    attach_groups(impl, "implementation")

    # Theory and implementation progress independently: a topic can be
    # closed before its block is written, a block done before its topic is
    # closed. The links between the two sides are information, never a
    # block. A link exists when the implementation side names the topic
    # (needs_theory, own or inherited from the step) or the theory side
    # names the stage (affects).
    def inherited_theory(stage):
        needs, cur, seen = list(stage["needs_theory"]), stage.get("parent"), set()
        while cur in impl and cur not in seen:
            seen.add(cur)
            needs += [t for t in impl[cur]["needs_theory"] if t not in needs]
            cur = impl[cur].get("parent")
        return needs

    for sid, stage in impl.items():
        links = inherited_theory(stage)
        links += [t for t, ts in theory.items() if sid in ts["affects"] and t not in links]
        stage["theory_links"] = [t for t in links if t in theory]
        stage["theory_open"] = [t for t in stage["theory_links"] if theory[t]["status"] != "closed"]
        # blocked_by: a dependency whose code is not in place (planned,
        # local, failing). Not "not done": done also needs the tester and
        # the cluster, and a block can be used long before that.
        blocked = [ref for ref in stage["depends_on"]
                   if ref in impl and impl[ref]["status"] in ("planned", "local", "failing")]
        stage["blocked_by"] = blocked if stage["has_own"] else []
    for sid, stage in theory.items():
        stage["blocked_by"] = [ref for ref in stage["depends_on"]
                               if ref in theory and theory[ref]["status"] != "closed"
                               ] if stage["has_own"] else []
        feeds = [i for i, s in impl.items() if sid in s["theory_links"]]
        stage["feeds"] = feeds

    # revisions: a decision on one side that the other side has to absorb
    attach_revisions(data, theory, impl, root)

    counts = {s: 0 for s in IMPL_STATES}
    for stage in impl.values():
        if stage["has_own"]:
            counts[stage["status"]] = counts.get(stage["status"], 0) + 1
    tcounts = {s: 0 for s in THEORY_STATES}
    for stage in theory.values():
        if stage["has_own"]:
            tcounts[stage["status"]] = tcounts.get(stage["status"], 0) + 1

    warnings = []
    if shallow:
        warnings.append("shallow clone: a block's last change may be reported "
                        "as the clone's oldest commit, which can turn done into "
                        "pushed; deepen with git fetch --depth=2000 origin %s" % branch)
    if spec_text is None:
        warnings.append("%s not found on %s" % (data["spec"], branch))
    unresolved = {s["id"]: s["unresolved_modules"] for s in impl.values()
                  if s.get("unresolved_modules")}
    if unresolved:
        warnings.append("module tokens of SPEC.md not found in the tree: %s"
                        % json.dumps(unresolved))

    return {
        "schema": SCHEMA_VERSION,
        "key": doc_key(data),
        "name": data["name"],
        "title": data.get("title", data["name"]),
        "repo": data.get("repo", ""),
        "branch": branch,
        "commit": head,
        "commit_short": head[:7],
        "commit_date": git(root, "log", "-1", "--format=%cI", head).strip(),
        "synced_at": now_iso(),
        "synced_from": source,
        "shallow_clone": shallow,
        "spec": data["spec"],
        "hpc_branch": data["hpc_branch"],
        "theory": [theory[s["id"]] for s in data["theory"]],
        "implementation": [impl[s["id"]] for s in data["implementation"]],
        "counts": {"implementation": counts, "theory": tcounts},
        "legend": {kind: [{"state": s, "color": c, "text": t, "meaning": m}
                          for s, c, t, m in states]
                   for kind, states in STATES.items()},
        "reports_seen": [{"path": r["path"], "commit": r["commit"][:7],
                          "date": r["date"], "verdict": r["verdict"]}
                         for r in reports],
        "hpc_markers_seen": len(markers),
        "warnings": warnings,
    }


# least advanced first; a group takes the colour of its least advanced member
AGGREGATE_ORDER = {
    "implementation": ["failing", "planned", "local", "pushed", "testing",
                       "tested", "running", "done"],
    "theory": ["open", "studying", "closed"],
}


def attach_groups(stages, kind):
    """Fill parent/children/depth and, for stages with children, aggregate.

    aggregate = {"worst": state, "counts": {state: n}} over every descendant
    that has a state of its own (has_own), the stage's own state included.
    A stage without a state of its own (an implementation group with no
    block) takes the aggregate as its status, so it can be drawn.
    """
    order = AGGREGATE_ORDER[kind]
    children = {sid: [] for sid in stages}
    for sid, stage in stages.items():
        parent = stage.get("parent")
        if parent in stages and parent != sid:
            children[parent].append(sid)
        else:
            stage["parent"] = None
    for sid, stage in stages.items():
        stage["children"] = children[sid]
        depth, cur, seen = 0, stage.get("parent"), set()
        while cur in stages and cur not in seen:
            seen.add(cur)
            depth += 1
            cur = stages[cur].get("parent")
        stage["depth"] = depth

    def own_states(sid, seen):
        if sid in seen:
            return []
        seen.add(sid)
        found = [stages[sid]["status"]] if stages[sid]["has_own"] else []
        for child in children[sid]:
            found += own_states(child, seen)
        return found

    for sid, stage in stages.items():
        if not children[sid]:
            stage["aggregate"] = None
            continue
        states = own_states(sid, set())
        counts = {}
        for state in states:
            counts[state] = counts.get(state, 0) + 1
        worst = min(states, key=lambda s: order.index(s) if s in order else -1) if states else None
        stage["aggregate"] = {"worst": worst, "counts": counts, "n": len(states)}
        if not stage["has_own"] and worst:
            stage["status"] = worst
            stage["reason"] = "group: least advanced member is %s" % worst


def attach_revisions(data, theory, impl, root):
    """Attach the pending revisions to their stages.

    A revision says: stage X has to change because of Y (a decision taken
    while studying a topic, or a finding made while implementing). For an
    implementation stage it stays pending until the stage's code changes
    after the flag (the flag records the branch commit) or until it is
    cleared by hand; for a theory stage, until the topic's status changes
    from what it was when flagged, or until cleared.
    """
    for stage in list(theory.values()) + list(impl.values()):
        stage["revisions"] = []
    for rev in data.get("revisions", []):
        sid = rev.get("stage")
        target = impl.get(sid) or theory.get(sid)
        if target is None or rev.get("cleared"):
            continue
        entry = {
            "from": str(rev.get("from") or ""),
            "on": str(rev.get("on") or ""),
            "decision": str(rev.get("decision") or ""),
            "note": str(rev.get("note") or ""),
            "at": str(rev.get("at") or "")[:7],
            "pending": True,
            "why_pending": "",
        }
        if sid in impl:
            last = target.get("last_change") or ""
            at = resolve_rev(root, rev.get("at")) if rev.get("at") else None
            if last and at and not is_ancestor(root, last, at):
                entry["pending"] = False   # the code changed after the flag
            else:
                entry["why_pending"] = ("no code change since %s" % entry["at"]
                                        if at else "pending until cleared")
        else:
            status_at = str(rev.get("status_at") or "")
            if status_at and target["status"] != status_at:
                entry["pending"] = False   # the topic moved on since the flag
            else:
                entry["why_pending"] = ("status still %s" % status_at if status_at
                                        else "pending until cleared")
        if entry["pending"]:
            target["revisions"].append(entry)
    for stage in list(theory.values()) + list(impl.values()):
        stage["revision_pending"] = bool(stage["revisions"])


def add_revision(path, stage_id, source, decision=None, note=None, root=None, status_now=None):
    """Append a revision flag to the yaml's top-level `revisions:` list."""
    data = load_roadmap(path)
    kinds = {s["id"]: "implementation" for s in data["implementation"]}
    kinds.update({s["id"]: "theory" for s in data["theory"]})
    if stage_id not in kinds:
        raise RoadmapError("stage %s not found in %s" % (stage_id, path))
    if source and source not in kinds:
        raise RoadmapError("source stage %s not found in %s" % (source, path))
    with open(path, "r", encoding="utf-8") as fh:
        text = fh.read()
    lines = text.splitlines(True)
    if lines and not lines[-1].endswith("\n"):
        lines[-1] += "\n"
    if not any(re.match(r"^revisions:\s*$", line) for line in lines):
        lines.append("\n# Revisions: a change one side has to absorb because of the other\n"
                     "# (roadmap.py revise). An implementation stage's flag clears when its\n"
                     "# code changes after the flag's commit; a topic's when its status\n"
                     "# changes; or by hand (revise --clear).\n")
        lines.append("revisions:\n")
    entry = ["  - stage: %s\n" % stage_id,
             "    from: %s\n" % (source or ""),
             "    on: %s\n" % dt.date.today().isoformat()]
    if kinds[stage_id] == "implementation":
        at = resolve_rev(root, data["branch"]) if root else None
        entry.append("    at: %s\n" % (at or ""))
    else:
        entry.append("    status_at: %s\n" % (status_now or ""))
    entry.append("    decision: %s\n" % (decision or ""))
    entry.append("    note: %s\n" % (yaml_scalar(note) if note else ""))
    entry.append("    cleared:\n")
    # the revisions list is the last top-level key: append at the end
    idx = max(i for i, line in enumerate(lines) if re.match(r"^revisions:\s*$", line))
    insert_at = len(lines)
    for j in range(idx + 1, len(lines)):
        if lines[j].strip() and not lines[j].startswith((" ", "#")):
            insert_at = j
            break
    lines[insert_at:insert_at] = entry
    with open(path, "w", encoding="utf-8", newline="\n") as fh:
        fh.write("".join(lines))


def clear_revisions(path, stage_id):
    """Set `cleared:` on every open revision of a stage; returns how many."""
    with open(path, "r", encoding="utf-8") as fh:
        lines = fh.read().splitlines(True)
    count, i = 0, 0
    while i < len(lines):
        if re.match(r"^  - stage:\s*%s\s*$" % re.escape(stage_id), lines[i]):
            j = i + 1
            cleared_line = None
            while j < len(lines) and lines[j].startswith("    "):
                if re.match(r"^    cleared:\s*$", lines[j]):
                    cleared_line = j
                j += 1
            if cleared_line is not None:
                lines[cleared_line] = "    cleared: %s\n" % dt.date.today().isoformat()
                count += 1
            i = j
        else:
            i += 1
    with open(path, "w", encoding="utf-8", newline="\n") as fh:
        fh.write("".join(lines))
    return count


def doc_key(data):
    repo = (data.get("repo") or "repo").split("/")[-1].lower()
    return "%s:%s" % (repo, data["name"])


def evaluate_stage(stage, data, root, head, remote_head, worktree_is_branch,
                   dirty, blocks, coverage, paths, reports, runs, markers,
                   now, stale_hours, last_finished):
    sid = stage["id"]
    result = {
        "id": sid,
        "title": stage.get("title", sid),
        "summary": stage.get("summary", ""),
        "blocks": stage["blocks"],
        "depends_on": as_list(stage.get("depends_on")),
        "needs_theory": as_list(stage.get("needs_theory")),
        "needs_hpc": bool(stage.get("needs_hpc")),
        "jobs": as_list(stage.get("jobs")),
        "note": stage.get("note", ""),
        "parent": stage.get("parent"),
        # a stage has a state of its own when it has a block or a manual
        # status; otherwise it is a pure group and takes its members' aggregate
        "has_own": bool(stage["blocks"]) or bool(stage.get("manual_status")),
        "modules": [],
        "unresolved_modules": [],
        "spec_status": "",
        "last_change": "",
        "last_change_date": "",
        "tester": {},
        "hpc": {},
        "status": "planned",
        "reason": "",
        "history": [],
    }

    # --- code: modules named by the SPEC entries of the stage's blocks
    tokens, spec_words, missing_blocks = [], {}, []
    for number in stage["blocks"]:
        block = blocks.get(number)
        if block is None:
            missing_blocks.append(number)
            continue
        tokens.extend(module_tokens(block))
        spec_words[number] = spec_status_word(block)
    modules, unresolved = resolve_modules(tokens, paths, coverage)
    result["modules"] = modules
    result["unresolved_modules"] = unresolved
    result["spec_status"] = ", ".join("%d: %s" % (n, w) for n, w in spec_words.items() if w)

    if stage.get("manual_status"):
        result["status"] = stage["manual_status"]
        result["manual"] = True
        result["reason"] = "set by hand: %s" % stage.get("manual_reason", "no reason given")
        return result
    if not stage["blocks"]:
        result["reason"] = ("step of the plan: no blocks defined yet "
                            "(roadmap.py add block ... --parent %s)" % sid)
        return result
    if missing_blocks and len(missing_blocks) == len(stage["blocks"]):
        result["reason"] = "no entry in SPEC.md for Block %s" % ", ".join(map(str, missing_blocks))
        return result
    drafted = [str(n) for n, w in spec_words.items() if w.startswith("drafted")]
    if drafted:
        result["reason"] = "SPEC.md status: drafted (Block %s)" % ", ".join(drafted)
        return result
    if not modules:
        result["reason"] = ("the modules of the SPEC entry do not exist on %s"
                            % data["branch"]) + (": %s" % ", ".join(unresolved)
                                                   if unresolved else "")
        return result

    # --- last change of the code, on the branch
    log = git(root, "log", "-1", "--format=%H%x09%cI", head, "--", *modules).strip()
    if log:
        result["last_change"], result["last_change_date"] = log.split("\t")
    last_change = result["last_change"] or head

    # --- local: not pushed
    if worktree_is_branch:
        touched = [p for p in dirty if any(claims(m, p) for m in modules)]
        if touched:
            result["status"] = "local"
            result["reason"] = "uncommitted changes in %s" % ", ".join(sorted(touched)[:3])
            return result
    if remote_head is None:
        result["status"] = "local"
        result["reason"] = "branch %s is not on origin" % data["branch"]
        return result
    if not is_ancestor(root, last_change, remote_head):
        result["status"] = "local"
        result["reason"] = "commit %s is not on origin/%s" % (last_change[:7], data["branch"])
        return result

    # --- HPC markers of this stage
    mine = [m for m in markers if marker_matches(m, sid, result["jobs"])]
    mine.sort(key=lambda m: (m.get("started") or ""), reverse=True)
    hpc = {}
    if mine:
        marker = mine[0]
        started = parse_time(marker.get("started"))
        age_h = ((now - started).total_seconds() / 3600.0) if started else None
        run_commit = resolve_rev(root, marker["commit"]) if marker.get("commit") else None
        hpc = {
            "state": marker.get("state", ""),
            "job": marker.get("job_id", ""),
            "job_name": marker.get("job_name", ""),
            "commit": (marker.get("commit") or "")[:7],
            "commit_known": run_commit is not None,
            "started": marker.get("started", ""),
            "ended": marker.get("ended", ""),
            "n_tasks": marker.get("n_tasks"),
            "n_failed": marker.get("n_failed"),
            "age_hours": (round(age_h, 1) if age_h is not None else None),
            "covers_current_code": bool(
                run_commit and is_ancestor(root, last_change, run_commit)),
            "stale": bool(marker.get("state") == "running" and age_h is not None
                          and age_h > stale_hours),
        }
    result["hpc"] = hpc
    if hpc.get("state") == "running" and not hpc.get("stale"):
        result["status"] = "running"
        result["reason"] = "HPC job %s running since %s" % (hpc["job"] or "?", hpc["started"])
        return result

    # --- tester
    covering = [r for r in reports
                if report_covers(r, stage["blocks"])
                and resolve_rev(root, r["commit"])
                and is_ancestor(root, last_change, r["commit"])]
    running = []
    for run in runs:
        if run["status"] != "running" or not resolve_rev(root, run["commit"]):
            continue
        if not is_ancestor(root, last_change, run["commit"]):
            continue
        in_scope = (run["mode"] == "full" or last_finished is None
                    or not resolve_rev(root, last_finished["commit"])
                    or not is_ancestor(root, last_change, last_finished["commit"]))
        if in_scope:
            running.append(run)
    tester = {}
    if covering:
        report = covering[0]
        open_findings = [f for f in report["findings"]
                         if f["severity"] in ("BLOCKER", "MAJOR")
                         and set(f["blocks"]) & set(stage["blocks"])
                         and not RESOLVED_WORDS.search(report["decisions"].get(f["id"], ""))]
        tester = {
            "report": report["path"], "commit": report["commit"][:7],
            "date": report["date"], "verdict": report["verdict"],
            "open_findings": [f["id"] + " " + f["severity"] for f in open_findings],
        }
    elif reports:
        older = [r for r in reports if report_covers(r, stage["blocks"])]
        if older:
            tester = {"report": older[0]["path"], "commit": older[0]["commit"][:7],
                      "date": older[0]["date"], "verdict": older[0]["verdict"],
                      "outdated": True}
    result["tester"] = tester
    if running:
        result["status"] = "testing"
        result["reason"] = "tester running on %s (%s)" % (running[0]["commit"][:7], running[0]["branch"])
        return result
    if tester.get("open_findings"):
        result["status"] = "failing"
        result["reason"] = "open %s in %s" % (", ".join(tester["open_findings"]), tester["report"])
        return result
    if hpc and hpc.get("state") == "failed" and hpc.get("covers_current_code"):
        result["status"] = "failing"
        result["reason"] = "HPC job %s failed (%s of %s tasks)" % (
            hpc["job"] or "?", hpc.get("n_failed"), hpc.get("n_tasks"))
        return result
    if hpc.get("stale"):
        result["status"] = "failing"
        result["reason"] = "HPC running marker is %.0f h old with no end marker" % hpc["age_hours"]
        return result
    if not covering:
        result["status"] = "pushed"
        result["reason"] = ("no tester report covers commit %s" % last_change[:7]
                            + (" (last report %s at %s)" % (tester["date"], tester["commit"])
                               if tester.get("outdated") else ""))
        return result
    if result["needs_hpc"] and not (hpc.get("state") == "done" and hpc.get("covers_current_code")):
        if hpc.get("state") == "done":
            why = ("at a commit this clone does not know" if not hpc.get("commit_known")
                   else "predates the last change")
        else:
            why = "missing"
        result["status"] = "tested"
        result["reason"] = "tester %s on %s; HPC run %s" % (
            tester["verdict"], tester["commit"], why)
        return result
    result["status"] = "done"
    result["reason"] = "tester %s on %s" % (tester["verdict"], tester["commit"]) + (
        "; HPC job %s done" % hpc["job"] if result["needs_hpc"] else "")
    return result


def marker_matches(marker, sid, jobs):
    if marker.get("stage") == sid:
        return True
    name = marker.get("job_name") or ""
    script = marker.get("script") or ""
    for pattern in jobs:
        for candidate in (name, script, os.path.basename(script)):
            if candidate and (candidate == pattern or fnmatch.fnmatchcase(candidate, pattern)):
                return True
    return False


def report_covers(report, numbers):
    if report["scope_blocks"] is None:
        return True
    return bool(set(numbers) & set(report["scope_blocks"]))


# ------------------------------------------------------------------ render ---

def mermaid(data, status):
    colors = {s: (fill, text) for s, fill, text, _ in STATES["implementation"]}
    colors.update({s: (fill, text) for s, fill, text, _ in STATES["theory"]})
    lines = ["flowchart LR"]
    for state, (fill, text) in colors.items():
        lines.append("    classDef %s fill:%s,color:%s,stroke:%s" % (state, fill, text, fill))
    lines.append("    classDef blocked stroke-dasharray:5 3,stroke:#333")
    styles = []

    def node_label(stage, kind):
        if kind == "theory":
            label = "%s %s" % (stage["id"], stage["title"])
            if stage["closed_on"]:
                label += "<br/>closed %s" % stage["closed_on"]
        else:
            blocks = ("Block %s " % ", ".join(map(str, stage["blocks"]))) if stage["blocks"] else ""
            label = "%s %s<br/>%s%s" % (stage["id"], stage["title"], blocks, stage["status"])
        if stage.get("revisions"):
            label += "<br/>REVISE: from %s" % ", ".join(r["from"] or "?" for r in stage["revisions"])
        return label

    def emit(stages, kind, parent, indent):
        pad = "    " * indent
        for stage in stages:
            if stage["parent"] != parent:
                continue
            if stage["children"]:
                agg = stage.get("aggregate") or {}
                counts = ", ".join("%d %s" % (n, s) for s, n in sorted(
                    agg.get("counts", {}).items(), key=lambda kv: -kv[1]))
                title = "%s %s" % (stage["id"], stage["title"])
                if counts:
                    title += " (" + counts + ")"
                lines.append('%ssubgraph grp_%s["%s"]' % (pad, stage["id"], esc(title)))
                lines.append("%s    direction TB" % pad)
                if stage["has_own"]:
                    lines.append('%s    %s["%s"]:::%s' % (
                        pad, stage["id"], esc(node_label(stage, kind)), stage["status"]))
                emit(stages, kind, stage["id"], indent + 1)
                lines.append("%send" % pad)
                worst = agg.get("worst") or stage["status"]
                styles.append("    style grp_%s stroke:%s,stroke-width:3px"
                              % (stage["id"], colors.get(worst, ("#8d99ae",))[0]))
            else:
                lines.append('%s%s["%s"]:::%s' % (
                    pad, stage["id"], esc(node_label(stage, kind)), stage["status"]))

    def endpoint(stage_map, sid):
        stage = stage_map.get(sid)
        if stage and stage["children"] and not stage["has_own"]:
            return "grp_" + sid
        return sid

    tmap = {s["id"]: s for s in status["theory"]}
    imap = {s["id"]: s for s in status["implementation"]}
    lines.append('    subgraph theory["Theory"]')
    lines.append("        direction TB")
    emit(status["theory"], "theory", None, 2)
    for stage in status["theory"]:
        for ref in stage["depends_on"]:
            lines.append("        %s --> %s" % (endpoint(tmap, ref), endpoint(tmap, stage["id"])))
    lines.append("    end")
    lines.append('    subgraph impl["Implementation"]')
    lines.append("        direction TB")
    emit(status["implementation"], "implementation", None, 2)
    for stage in status["implementation"]:
        for ref in stage["depends_on"]:
            lines.append("        %s --> %s" % (endpoint(imap, ref), endpoint(imap, stage["id"])))
    lines.append("    end")
    drawn = set()
    for stage in status["implementation"]:
        # own links only: a link inherited from the step is drawn at the step
        own = list(stage["needs_theory"]) + [t for t in stage.get("theory_links", [])
                                            if stage["id"] in tmap.get(t, {}).get("affects", [])]
        for ref in own:
            edge = (endpoint(tmap, ref), endpoint(imap, stage["id"]))
            if ref in tmap and edge not in drawn:
                drawn.add(edge)
                lines.append("    %s -.-> %s" % edge)
    blocked = [s["id"] for s in status["theory"] + status["implementation"]
               if s.get("blocked_by") and (s["has_own"] or not s["children"])]
    if blocked:
        lines.append("    class %s blocked" % ",".join(blocked))
    lines.extend(styles)
    return "\n".join(lines)


def esc(text):
    return text.replace('"', "#quot;")


def render_markdown(data, status):
    out = []
    out.append("# Roadmap -- %s" % status["title"])
    out.append("")
    out.append("Generated by `specs/roadmap/roadmap.py render` from `specs/roadmap/%s.yaml`; "
               "do not edit. Branch `%s` at `%s` (%s); synced %s from %s."
               % (status["name"], status["branch"], status["commit_short"],
                  status["commit_date"][:10], status["synced_at"], status["synced_from"]))
    out.append("")
    out.append("```mermaid")
    out.append(mermaid(data, status))
    out.append("```")
    out.append("")
    out.append("## Legend")
    out.append("")
    out.append("| State | Colour | Meaning |")
    out.append("|---|---|---|")
    for kind in ("implementation", "theory"):
        for state, color, _, meaning in STATES[kind]:
            out.append("| %s (%s) | %s | %s |" % (state, kind, color, meaning))
    out.append("")
    out.append("A dashed border marks a stage whose dependency is not in place (a "
               "block that is planned, local or failing; a topic not closed). Theory "
               "and implementation progress independently: the dotted edges between "
               "them are links, not conditions. REVISE marks a stage that has to "
               "change because of a decision taken on the other side.")
    out.append("")
    out.append("## Theory")
    out.append("")
    out.append("| Stage | Group | Status | Closed | Decision | Depends on | Linked blocks | Revise | Summary |")
    out.append("|---|---|---|---|---|---|---|---|---|")
    for stage in status["theory"]:
        out.append("| %s %s | %s | %s | %s | %s | %s | %s | %s | %s |" % (
            stage["id"], stage["title"], stage["parent"] or "", stage["status"],
            stage["closed_on"], stage["decision"], ", ".join(stage["depends_on"]),
            ", ".join(stage["feeds"]), revise_cell(stage), cell(stage["summary"])))
    out.append("")
    out.append("## Implementation")
    out.append("")
    out.append("| Stage | Group | Blocks | Status | Why | Last change | Tester | HPC | Depends on | Theory (open) | Revise | Summary |")
    out.append("|---|---|---|---|---|---|---|---|---|---|---|---|")
    for stage in status["implementation"]:
        tester = stage["tester"]
        hpc = stage["hpc"]
        theory_cell = ", ".join(t + (" (open)" if t in stage.get("theory_open", []) else "")
                                for t in stage.get("theory_links", []))
        out.append("| %s %s | %s | %s | %s | %s | %s | %s | %s | %s | %s | %s | %s |" % (
            stage["id"], stage["title"], stage["parent"] or "",
            ", ".join(map(str, stage["blocks"])),
            stage["status"], cell(stage["reason"]),
            (stage["last_change"][:7] + " " + stage["last_change_date"][:10]).strip(),
            ((tester.get("verdict", "") + " " + tester.get("commit", "")).strip()
             + (" (outdated)" if tester.get("outdated") else "")) if tester else "",
            ((hpc.get("state", "") + " " + (hpc.get("job") or "")).strip()) if hpc else "",
            ", ".join(stage["depends_on"]), theory_cell, revise_cell(stage), cell(stage["summary"])))
    out.append("")
    return "\n".join(out) + "\n"


def revise_cell(stage):
    return "; ".join("from %s%s%s" % (r["from"] or "?", (" " + r["decision"]) if r["decision"] else "",
                                      (": " + r["note"]) if r["note"] else "")
                     for r in stage.get("revisions", [])).replace("|", "/")


def cell(text):
    return str(text or "").replace("|", "/").replace("\n", " ")


# -------------------------------------------------------------------- set ---

def set_theory_status(path, stage_id, new_status, note=None, decision=None, on=None):
    """Edit the status of a theory stage in place, keeping the file's layout."""
    if new_status not in THEORY_STATES:
        raise RoadmapError("status must be one of %s" % THEORY_STATES)
    with open(path, "r", encoding="utf-8") as fh:
        lines = fh.read().splitlines(True)
    # find the stage block: a line "- id: <stage_id>" inside the theory list
    start = None
    for i, line in enumerate(lines):
        if re.match(r"^\s*-\s*id:\s*['\"]?%s['\"]?\s*$" % re.escape(stage_id), line):
            start = i
            break
    if start is None:
        raise RoadmapError("stage %s not found in %s" % (stage_id, path))
    indent = len(lines[start]) - len(lines[start].lstrip())
    end = len(lines)
    for j in range(start + 1, len(lines)):
        stripped = lines[j].lstrip()
        cur = len(lines[j]) - len(stripped)
        if stripped and cur <= indent and not stripped.startswith("#"):
            end = j
            break
    field_indent = " " * (indent + 2)
    on = on or dt.date.today().isoformat()
    updates = {"status": new_status}
    if new_status == "closed":
        updates["closed_on"] = on
    if decision:
        updates["decision"] = decision
    if note:
        updates["note"] = note
    present = {}
    for j in range(start + 1, end):
        match = re.match(r"^(\s*)(\w+):", lines[j])
        if match:
            present[match.group(2)] = j
    for key, value in updates.items():
        text = "%s%s: %s\n" % (field_indent, key, yaml_scalar(value))
        if key in present:
            lines[present[key]] = text
        else:
            lines.insert(end, text)
            end += 1
    if new_status != "closed" and "closed_on" in present:
        lines[present["closed_on"]] = "%sclosed_on:\n" % field_indent
    with open(path, "w", encoding="utf-8", newline="\n") as fh:
        fh.write("".join(lines))


def yaml_scalar(value):
    text = str(value)
    if re.match(r"^[A-Za-z0-9_.:-]+$", text):
        return text
    return json.dumps(text)


# -------------------------------------------------------------------- add ---

def add_stage(path, kind, sid, fields):
    """Append a stage entry to the theory or implementation list of the yaml.

    kind: "topic" (theory list), "block" or "step" (implementation list).
    fields: ordered (key, value) pairs; a None value is written as an empty
    field, a list as [a, b]. The entry goes at the end of its list, so the
    rest of the file (comments included) is untouched. The parent, the
    dependencies and the id are validated against the file first.
    """
    data = load_roadmap(path)
    pool = data["theory"] if kind == "topic" else data["implementation"]
    other = data["implementation"] if kind == "topic" else data["theory"]
    ids = {s["id"] for s in pool}
    if not re.match(r"^[A-Za-z][A-Za-z0-9_.-]*$", sid):
        raise RoadmapError("id %r: letters, digits, _ . - only, starting with a letter" % sid)
    if sid in ids or sid in {s["id"] for s in other}:
        raise RoadmapError("id %s already exists in %s" % (sid, path))
    parent = dict(fields).get("parent")
    if parent and parent not in ids:
        raise RoadmapError("parent %s: no such %s stage" % (parent, "theory" if kind == "topic" else "implementation"))
    for ref in as_list(dict(fields).get("depends_on")):
        if ref not in ids:
            raise RoadmapError("depends_on %s: no such stage" % ref)
    if kind != "topic":
        tids = {s["id"] for s in data["theory"]}
        for ref in as_list(dict(fields).get("needs_theory")):
            if ref not in tids:
                raise RoadmapError("needs_theory %s: no such theory stage" % ref)
    if kind == "block":
        used = {b: s["id"] for s in data["implementation"] for b in s["blocks"]}
        wanted = as_list(dict(fields).get("blocks")) + as_list(dict(fields).get("block"))
        for b in wanted:
            if int(b) in used:
                raise RoadmapError("Block %s already belongs to %s" % (b, used[int(b)]))

    with open(path, "r", encoding="utf-8") as fh:
        text = fh.read()
    lines = text.splitlines(True)
    entry = ["  - id: %s\n" % sid]
    for key, value in fields:
        if value is None or value == "":
            entry.append("    %s:\n" % key)
        elif isinstance(value, list):
            entry.append("    %s: [%s]\n" % (key, ", ".join(yaml_scalar(v) for v in value)))
        elif isinstance(value, bool):
            entry.append("    %s: %s\n" % (key, "true" if value else "false"))
        else:
            entry.append("    %s: %s\n" % (key, yaml_scalar(value)))
    if kind == "topic":
        # before the top-level "implementation:" line
        idx = next((i for i, line in enumerate(lines) if re.match(r"^implementation:\s*$", line)), None)
        if idx is None:
            raise RoadmapError("no top-level 'implementation:' key in %s" % path)
        # keep one blank line between the lists
        insert_at = idx
        while insert_at > 0 and not lines[insert_at - 1].strip():
            insert_at -= 1
        lines[insert_at:insert_at] = entry
    else:
        if lines and not lines[-1].endswith("\n"):
            lines[-1] += "\n"
        lines.extend(entry)
    with open(path, "w", encoding="utf-8", newline="\n") as fh:
        fh.write("".join(lines))
    problems = check_roadmap(load_roadmap(path))
    if problems:
        # roll back rather than leave a broken file
        with open(path, "w", encoding="utf-8", newline="\n") as fh:
            fh.write(text)
        raise RoadmapError("not added: " + "; ".join(problems))


def split_list(value):
    return [v.strip() for v in (value or "").split(",") if v.strip()]


# ------------------------------------------------------------------- init ---

def load_plan(path):
    need_yaml()
    with open(path, "r", encoding="utf-8") as fh:
        plan = yaml.safe_load(fh) or {}
    steps = plan.get("steps") or []
    if not steps:
        raise RoadmapError("%s has no steps" % path)
    seen = set()
    for k, step in enumerate(steps, start=1):
        step.setdefault("id", "S%d" % k)
        if not re.match(r"^[A-Za-z][A-Za-z0-9_.-]*$", str(step["id"])) or step["id"] in seen:
            raise RoadmapError("plan step %r: invalid or duplicate id" % step.get("id"))
        seen.add(step["id"])
        if not step.get("title"):
            raise RoadmapError("plan step %s has no title" % step["id"])
        step["summary"] = step.get("summary") or ""
        step["theory"] = step.get("theory") or ""
        step["theory_title"] = step.get("theory_title") or ("Theory of " + step["title"])
        step["depends_on"] = as_list(step.get("depends_on"))
        step["needs_hpc"] = bool(step.get("needs_hpc"))
        step["blocks"] = [int(b) for b in as_list(step.get("blocks"))]
        step["theory_id"] = (step.get("theory_id") or ("T" + step["id"])) if (
            step["theory"] or step.get("theory_id")) else None
    for step in steps:
        for ref in step["depends_on"]:
            if ref not in seen:
                raise RoadmapError("plan step %s depends on unknown step %s" % (step["id"], ref))
    return steps


def init_roadmap(root, name, branch, repo=None, spec="specs/SPEC.md", out=None, title=None,
                 plan=None):
    head = resolve_rev(root, branch) or resolve_rev(root, "origin/" + branch)
    if head is None:
        raise GitError("branch %s not found" % branch)
    spec_text = read_at(root, head, spec) or ""
    blocks = parse_spec(spec_text)
    title = title or (re.match(r"^#\s*SPEC\s*--\s*(.*)$", spec_text.splitlines()[0]).group(1)
                      if spec_text and re.match(r"^#\s*SPEC\s*--", spec_text) else name)
    if repo is None:
        url = git(root, "remote", "get-url", "origin", check=False).strip()
        match = re.search(r"github\.com[:/]([^/]+/[^/.]+)", url)
        repo = match.group(1) if match else ""
    steps = load_plan(plan) if plan else None
    lines = [
        "# Roadmap of the subproject %s. Source of truth for the stages and the" % name,
        "# theory statuses; the implementation statuses are computed by",
        "# specs/roadmap/roadmap.py (never written by hand).",
        "#",
    ]
    if steps:
        lines += [
            "# Drafted by `roadmap.py init --plan`: one implementation step S<k> per",
            "# plan step (a group; its blocks are added while the step is carried out,",
            "# with `roadmap.py add block`) and one theory group TS<k> per step (its",
            "# topics with `roadmap.py add topic`). Lines marked DRAFT are guesses.",
        ]
    else:
        lines += [
            "# Drafted by `roadmap.py init` from %s: review every stage, the" % spec,
            "# dependencies and needs_hpc, and replace the theory stages with the",
            "# topics you actually have to study. Lines marked DRAFT are guesses.",
        ]
    lines += [
        "schema: %d" % SCHEMA_VERSION,
        "name: %s" % name,
        "title: %s" % yaml_scalar(title),
        "repo: %s" % repo,
        "branch: %s" % branch,
        "spec: %s" % spec,
        "hpc_branch: hpc-status",
        "stale_after_hours: 48      # a running HPC marker older than this is reported as failing",
        "",
        "theory:",
    ]
    section_blocks = {}
    block_parent = {}
    if steps:
        # a theory group for the steps that rest on concepts the plan names
        # (`theory:`); routines get none. Topics come later (add topic).
        for step in steps:
            if not step["theory_id"]:
                continue
            lines += [
                "  - id: %s" % step["theory_id"],
                "    title: %s" % yaml_scalar(step["theory_title"]),
                "    summary: %s" % yaml_scalar(step["theory"] or ("DRAFT: what has to be understood for " + step["title"])),
                "    refs: []",
                "    depends_on: []",
                "    # no status line: a group, coloured by its topics (add topic ... --parent %s)" % step["theory_id"],
            ]
        for step in steps:
            for number in step["blocks"]:
                block_parent[number] = step["id"]
    else:
        # theory stages drafted from the subsections of the mathematical
        # formulation ("### 2.k Title (Blocks n, m; ...)"): one stage each,
        # fed to the blocks its heading names
        sections = re.findall(r"^###\s+2\.(\d+)\s+(.*?)\s*$", spec_text, re.MULTILINE)
        if not sections:
            sections = [("1", "DRAFT: first topic to study")]
        for k, (num, heading) in enumerate(sections, start=1):
            sid = "T%d" % k
            # the blocks a section covers sit in its parentheses:
            # "### 2.2 Per-node measurement (Blocks 3, 5; handoff Eqs. 1-5)"
            section_blocks[sid] = block_refs(heading.replace("(", " ").replace(")", " "))
            short = re.sub(r"\s*\(.*?\)\s*$", "", heading)
            lines += [
                "  - id: %s" % sid,
                "    title: %s" % yaml_scalar(short),
                "    summary: \"DRAFT: from SPEC section 2.%s\"" % num,
                "    refs: []",
                "    parent:                 # id of the group stage this belongs to, if any",
                "    depends_on: %s" % ("[]" if k == 1 else "[T%d]" % (k - 1)),
                "    status: open",
                "    closed_on:",
                "    decision:",
            ]
    lines.append("")
    lines.append("implementation:")
    if steps:
        lines.append("  # ---- steps of the plan (groups; blocks are added under them as they are written)")
        for step in steps:
            lines += [
                "  - id: %s" % step["id"],
                "    title: %s" % yaml_scalar(step["title"]),
                "    summary: %s" % yaml_scalar(step["summary"] or ("DRAFT: " + step["title"])),
                "    depends_on: [%s]" % ", ".join(step["depends_on"]),
                "    needs_theory: [%s]" % (step["theory_id"] or ""),
                "    needs_hpc: %s" % ("true" if step["needs_hpc"] else "false"),
            ]
        if blocks:
            lines.append("")
            lines.append("  # ---- blocks already in SPEC.md (parent = the step each belongs to)")
    for number in sorted(blocks):
        block = blocks[number]
        consumed = sorted(n for n in consumed_blocks(block) if n in blocks and n != number)
        # a stage depends on the EARLIER blocks it consumes; a dependency on
        # a later block (a feedback loop in the data flow) is dropped so the
        # roadmap stays a DAG, and noted for review
        deps = [n for n in consumed if n < number]
        dropped = [n for n in consumed if n > number]
        needs_hpc = bool(HPC_WORDS.search(block["text"]))
        header = block["name"]
        needs_theory = [sid for sid, nums in section_blocks.items() if number in nums]
        short = re.sub(r"\s*\(.*$", "", header).strip()
        parent = block_parent.get(number)
        lines += [
            "  - id: I%d" % number,
            "    title: %s" % yaml_scalar(short),
            "    summary: %s" % yaml_scalar("DRAFT: " + short),
            "    block: %d" % number,
            ("    parent: %s" % parent) if parent else
            "    parent:                 # DRAFT: the step (or group) this block belongs to",
            "    depends_on: [%s]" % ", ".join("I%d" % n for n in deps)
            + ("   # DRAFT: also consumes later Block %s (loop), not drawn"
               % ", ".join(map(str, dropped)) if dropped else ""),
            "    needs_theory: [%s]" % ", ".join(sorted(set(needs_theory))),
            "    needs_hpc: %s" % ("true" if needs_hpc else "false")
            + ("      # DRAFT: the entry mentions the cluster" if needs_hpc else ""),
            "    jobs: []                # job script paths or PBS/SLURM job names that run this stage",
        ]
    text = "\n".join(lines) + "\n"
    out = out or os.path.join(root, "specs", "roadmap", name + ".yaml")
    os.makedirs(os.path.dirname(out), exist_ok=True)
    if os.path.exists(out):
        raise RoadmapError("%s exists; delete it first or choose another --name" % out)
    with open(out, "w", encoding="utf-8", newline="\n") as fh:
        fh.write(text)
    # install the tool beside the roadmap and ignore the cluster's state dir
    here = os.path.dirname(os.path.abspath(__file__))
    for fname in ("roadmap.py", "hpc_status.sh"):
        src, dst = os.path.join(here, fname), os.path.join(os.path.dirname(out), fname)
        if os.path.abspath(src) != os.path.abspath(dst) and os.path.exists(src):
            shutil.copyfile(src, dst)
    gitignore = os.path.join(root, ".gitignore")
    marker = ".roadmap/"
    existing = open(gitignore, encoding="utf-8").read() if os.path.exists(gitignore) else ""
    if marker not in existing.splitlines():
        with open(gitignore, "a", encoding="utf-8", newline="\n") as fh:
            fh.write(("" if existing.endswith("\n") or not existing else "\n")
                     + "# roadmap: the cluster-side state of hpc_status.sh\n" + marker + "\n")
    return out


# ------------------------------------------------------------------- main ---

def find_root(args):
    if getattr(args, "root", None):
        return os.path.abspath(args.root)
    if getattr(args, "yaml", None):
        root = repo_root_of(args.yaml)
        if root:
            return root
    return os.path.abspath(os.getcwd())


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    sub = parser.add_subparsers(dest="command", required=True)

    p = sub.add_parser("init", help="draft a roadmap from a plan and/or SPEC.md")
    p.add_argument("--name", required=True)
    p.add_argument("--branch", required=True)
    p.add_argument("--repo", default=None, help="owner/name on GitHub")
    p.add_argument("--plan", default=None, help="PLAN.yaml with the coarse steps (see the module docstring)")
    p.add_argument("--spec", default="specs/SPEC.md")
    p.add_argument("--title", default=None)
    p.add_argument("--out", default=None)
    p.add_argument("--root", default=None)

    p = sub.add_parser("add", help="add a block, a topic or a step to a roadmap")
    p.add_argument("--yaml", required=True)
    p.add_argument("what", choices=["block", "topic", "step"])
    p.add_argument("id")
    p.add_argument("--title", required=True)
    p.add_argument("--summary", default=None)
    p.add_argument("--parent", default=None, help="the step (block, topic) or group this belongs to")
    p.add_argument("--block", default=None, help="SPEC block number(s), comma-separated (block only)")
    p.add_argument("--depends-on", default=None, help="comma-separated ids of the same kind")
    p.add_argument("--needs-theory", default=None, help="comma-separated theory ids (block, step)")
    p.add_argument("--needs-hpc", action="store_true")
    p.add_argument("--jobs", default=None, help="comma-separated job names or script paths (block)")
    p.add_argument("--refs", default=None, help="comma-separated references (topic)")
    p.add_argument("--status", default="open", choices=THEORY_STATES, help="topic only")
    p.add_argument("--theory", default=None,
                   help="step only: id of a theory group to create for the step's concepts "
                        "(default T<step id> when --theory-summary is given; none otherwise)")
    p.add_argument("--theory-summary", default=None,
                   help="step only: the concepts (mathematics, physics, biology, engineering) "
                        "the step rests on; creates the theory group")

    p = sub.add_parser("revise", help="flag a stage that has to change because of another")
    p.add_argument("--yaml", required=True)
    p.add_argument("stage", help="the stage that has to change (implementation or theory)")
    p.add_argument("--from", dest="source", default=None, help="the stage whose outcome requires it")
    p.add_argument("--decision", default=None)
    p.add_argument("--note", default=None)
    p.add_argument("--clear", action="store_true", help="clear the stage's open revisions instead")
    p.add_argument("--root", default=None)

    p = sub.add_parser("check", help="validate a roadmap file")
    p.add_argument("--yaml", required=True)
    p.add_argument("--root", default=None)

    p = sub.add_parser("status", help="compute the stage statuses")
    p.add_argument("--yaml", required=True)
    p.add_argument("--root", default=None)
    p.add_argument("--json", default=None, help="write the status document here")
    p.add_argument("--no-fetch", action="store_true")
    p.add_argument("--source", default="chat")

    p = sub.add_parser("render", help="write the Mermaid markdown")
    p.add_argument("--yaml", required=True)
    p.add_argument("--root", default=None)
    p.add_argument("--status", default=None, help="a status JSON (default: compute)")
    p.add_argument("--out", default=None)
    p.add_argument("--no-fetch", action="store_true")

    p = sub.add_parser("set", help="set a theory stage's status")
    p.add_argument("--yaml", required=True)
    p.add_argument("stage")
    p.add_argument("status", choices=THEORY_STATES)
    p.add_argument("--note", default=None)
    p.add_argument("--decision", default=None)
    p.add_argument("--on", default=None, help="date, default today")

    args = parser.parse_args(argv)
    try:
        return run(args)
    except GitError as exc:
        print("error: %s" % exc, file=sys.stderr)
        return 2
    except RoadmapError as exc:
        print("error: %s" % exc, file=sys.stderr)
        return 1


def run(args):
    root = find_root(args)
    if args.command == "init":
        out = init_roadmap(root, args.name, args.branch, args.repo, args.spec,
                           args.out, args.title, args.plan)
        print("wrote %s" % out)
        data = load_roadmap(out)
        for problem in check_roadmap(data):
            print("warning: %s" % problem)
        return 0
    if args.command == "add":
        return run_add(args)
    if args.command == "revise":
        if args.clear:
            n = clear_revisions(args.yaml, args.stage)
            print("%d revision(s) of %s cleared" % (n, args.stage))
            return 0
        data = load_roadmap(args.yaml)
        status_now = next((s["status"] for s in data["theory"] if s["id"] == args.stage), None)
        add_revision(args.yaml, args.stage, args.source, args.decision, args.note,
                     root=root, status_now=status_now)
        print("%s flagged for revision%s" % (args.stage, (" from %s" % args.source) if args.source else ""))
        return 0
    data = load_roadmap(args.yaml)
    if args.command == "set":
        set_theory_status(args.yaml, args.stage, args.status, args.note,
                          args.decision, args.on)
        print("%s -> %s" % (args.stage, args.status))
        return 0
    problems = check_roadmap(data)
    if args.command == "check":
        for problem in problems:
            print("problem: %s" % problem)
        print("ok" if not problems else "%d problem(s)" % len(problems))
        return 0 if not problems else 1
    if problems:
        raise RoadmapError("; ".join(problems))
    if args.command == "status":
        status = compute_status(data, root, fetch=not args.no_fetch, source=args.source)
        text = json.dumps(status, indent=2, sort_keys=False)
        if args.json:
            with open(args.json, "w", encoding="utf-8", newline="\n") as fh:
                fh.write(text + "\n")
            print("wrote %s: %s" % (args.json, summary(status)))
        else:
            print(text)
        return 0
    if args.command == "render":
        if args.status:
            with open(args.status, encoding="utf-8") as fh:
                status = json.load(fh)
        else:
            status = compute_status(data, root, fetch=not args.no_fetch)
        out = args.out or os.path.splitext(args.yaml)[0] + ".md"
        with open(out, "w", encoding="utf-8", newline="\n") as fh:
            fh.write(render_markdown(data, status))
        print("wrote %s: %s" % (out, summary(status)))
        return 0
    return 1


def run_add(args):
    if args.what == "block":
        if not args.block:
            raise RoadmapError("add block needs --block N (the SPEC block number)")
        if not args.parent:
            raise RoadmapError("add block needs --parent STEP (the step or group it belongs to)")
        blocks = [int(b) for b in split_list(args.block)]
        fields = [
            ("title", args.title),
            ("summary", args.summary or ("DRAFT: " + args.title)),
            ("blocks" if len(blocks) > 1 else "block", blocks if len(blocks) > 1 else blocks[0]),
            ("parent", args.parent),
            ("depends_on", split_list(args.depends_on)),
            ("needs_theory", split_list(args.needs_theory)),
            ("needs_hpc", bool(args.needs_hpc)),
            ("jobs", split_list(args.jobs)),
        ]
        add_stage(args.yaml, "block", args.id, fields)
        print("added block stage %s (Block %s) under %s" % (args.id, args.block, args.parent))
        return 0
    if args.what == "topic":
        if not args.parent:
            raise RoadmapError("add topic needs --parent TSTEP (the theory group it belongs to)")
        fields = [
            ("title", args.title),
            ("summary", args.summary or ("DRAFT: " + args.title)),
            ("refs", split_list(args.refs)),
            ("parent", args.parent),
            ("depends_on", split_list(args.depends_on)),
            ("status", args.status),
            ("closed_on", dt.date.today().isoformat() if args.status == "closed" else None),
            ("decision", None),
        ]
        add_stage(args.yaml, "topic", args.id, fields)
        print("added topic %s (%s) under %s" % (args.id, args.status, args.parent))
        return 0
    # step: an implementation group plus, by default, a paired theory group
    # a theory group only when the step rests on concepts the user named
    # (--theory-summary, or --theory <id>); routines get none
    theory_id = args.theory or (("T" + args.id) if args.theory_summary else "none")
    needs = split_list(args.needs_theory)
    if theory_id.lower() != "none":
        # a sub-step's theory group nests under the parent step's, when that exists
        theory_ids = {s["id"] for s in load_roadmap(args.yaml)["theory"]}
        theory_parent = ("T" + args.parent) if args.parent and ("T" + args.parent) in theory_ids else None
        add_stage(args.yaml, "topic", theory_id, [
            ("title", "Theory of " + args.title),
            ("summary", args.theory_summary or ("DRAFT: what has to be understood for " + args.title)),
            ("refs", []),
            ("parent", theory_parent),
            ("depends_on", []),
        ])
        needs = needs + [theory_id]
    add_stage(args.yaml, "step", args.id, [
        ("title", args.title),
        ("summary", args.summary or ("DRAFT: " + args.title)),
        ("parent", args.parent),
        ("depends_on", split_list(args.depends_on)),
        ("needs_theory", needs),
        ("needs_hpc", bool(args.needs_hpc)),
    ])
    print("added step %s%s" % (args.id, (" with theory group %s" % theory_id) if theory_id.lower() != "none" else ""))
    return 0


def summary(status):
    parts = ["%s %d" % (s, n) for s, n in status["counts"]["implementation"].items() if n]
    tparts = ["%s %d" % (s, n) for s, n in status["counts"]["theory"].items() if n]
    return "implementation " + ", ".join(parts) + "; theory " + ", ".join(tparts)


if __name__ == "__main__":
    sys.exit(main())
