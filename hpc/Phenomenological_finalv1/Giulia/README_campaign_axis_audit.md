# campaign_axis_audit -- what was swept, with which seeds, in which culture

Read-only auditor for a cohort of sweep campaigns produced by
`HPC_main_sweep.py`. Answers, per campaign and across campaigns:

1. Which parameters were SWEPT -- declared AND verified against the data.
2. Which BOUNDS each campaign used, aligned BY AXIS NAME.
3. The SEED footprint, with within-campaign reuse and pairwise
   cross-campaign overlap (a direct measurement of the collision bug).
4. The CULTURE GEOMETRY and the connectivity prior actually drawn.

Files:

    campaign_axis_audit.py               the auditor
    smoke_test_campaign_axis_audit.py    54-check correctness harness


## Why axis names, not indices

The parameter registry changed dimensionality between campaigns. The copy in
`hpc/Phenomenological_finalv1/` is 36-D (`assert PARAM_BOUNDS.shape == (36, 2)`,
no `O_N`); the `Giulia/` copy -- which matches `Giulia_Astro/` on the cluster --
is 37-D with `O_N` at index 36. Comparing `params[j]` across campaigns is
therefore silently wrong. Everything here is keyed by name.


## Why declared AND empirical

The declaration (`job_args.json:_axis_declaration`) says what the launcher
intended; the `iter_*.npz` `params` columns say what the sampler actually did.
They disagree in two ways that matter:

  * an axis declared swept whose column is constant (`swept-but-const`), and
  * an axis that varies but is causally INERT -- drawn, recorded, read by
    nothing.

A variance scan alone CANNOT detect the second case: the column varies
perfectly well. That is why the declaration is read too. Under
`MODE=Neuronal` no `Astrocyte_Group` / `Gliotransmission` / `Synapse_to_astro`
object is instantiated, so every astrocyte axis drawn in that mode is inert.
For campaigns predating `axis_declaration` (manifest_version < 4) the tool
infers inertness from `mode` and says so in the report.


## Guarantees

  * READ-ONLY. Nothing under the campaign tree is created, modified or
    deleted. All output goes to `--out`. The smoke test proves this with a
    byte-for-byte tree digest taken before and after a full run.
  * stdlib + numpy only. No Brian2, no scipy, no matplotlib. The registry
    sources are parsed with `ast`, never imported, so the tool is safe on
    files that import Brian2.
  * `manifest.json` is read HEAD-ONLY (everything before the `"topologies"`
    array), so a multi-hundred-MB task manifest costs a few KB.


## Cluster-side verification block

Paste this ONCE after transfer, before running anything. Line endings first:
a broken shebang means the interpreter is never reached, so the later checks
would never run.

    python3 --version; echo "conda env: ${CONDA_DEFAULT_ENV:-none}"

    python3 - <<'EOF'
    import sys
    bad = []
    for p in ("campaign_axis_audit.py", "smoke_test_campaign_axis_audit.py"):
        b = [hex(c) for c in open(p, "rb").read() if c > 127]
        n = open(p, "rb").read().count(b"\r")
        if b or n:
            bad.append((p, b[:6], n))
    for p, b, n in bad:
        print("CORRUPT", p, "non-ascii:", b, "CR bytes:", n)
    print("FAIL: %d file(s)" % len(bad) if bad
          else "OK: pure ASCII, LF-only")
    sys.exit(1 if bad else 0)
    EOF

    python3 -m py_compile campaign_axis_audit.py smoke_test_campaign_axis_audit.py && echo "OK: compiles"
    python3 smoke_test_campaign_axis_audit.py | tail -3

Expected last line:

    ALL 8 TEST GROUPS PASSED (54 checks)

If the smoke test fails, do NOT run the auditor against real campaigns --
the failure is either transfer corruption or a numpy version difference, and
either one makes the audit output untrustworthy.


## Running it

Login node, no allocation needed. Metadata-only first (seconds, opens no
`.npz`):

    python3 campaign_axis_audit.py /davinci-1/home/ldellamea/ANN/Phenomenological/Main/Giulia_Astro --glob 'campaign_cadex_hhgap_v*' --quick --out ./audit_hhgap_quick

Expect one `[audit] scanning ...` line per campaign found (4: v1, v2, v4, v5
-- v3 does not exist, so the glob simply will not match it), then a summary
block per campaign giving `tasks=`, `mode=`, `sweep_group=`, the culture
line, `swept axes: N (astro M, of which inert K)`, and the seed range. If a
campaign reports `tasks=0`, it is empty or never started.

Then the full pass, with the registry comparison enabled:

    python3 campaign_axis_audit.py /davinci-1/home/ldellamea/ANN/Phenomenological/Main/Giulia_Astro --glob 'campaign_cadex_hhgap_v*' --registry-src /davinci-1/home/ldellamea/ANN/Phenomenological/Main/Giulia_Astro/HPC_single_run.py --sweep-src /davinci-1/home/ldellamea/ANN/Phenomenological/Main/Giulia_Astro/HPC_main_sweep.py --out ./audit_hhgap

Outputs under `--out`:

    campaign_axis_audit.md      the report (9 sections)
    campaign_axis_audit.json    the same content, machine-readable

The two lines that answer the seed question directly are in the stdout
summary:

    campaign_cadex_hhgap_vA   vs campaign_cadex_hhgap_vB   : <n>  <-- COLLISION
    campaign_cadex_hhgap_vA   vs campaign_cadex_hhgap_vC   : <n>  <-- CONTAINMENT

`CONTAINMENT` is the severe case: one campaign's seed set is a subset of the
other's, i.e. its simulations are duplicates wherever the code was also
unchanged. A zero on every pair means the seed ranges were disjoint.


## Useful flags

    --quick                    metadata only; open no iter_*.npz
    --iters-per-campaign N     npz sampled per campaign (default 300)
    --topos-per-task N         topologies sampled per task (default 3)
    --iters-per-topo N         iters sampled per topology (default 4)
    --max-tasks N              cap task dirs read per campaign (0 = all)
    --campaigns A B C          explicit campaign dirs, overrides --glob
    --seed N                   RNG seed for file sampling (default 0)
    --manifest-head-bytes N    bytes read before the topologies array
    --max-manifest-mb N        refuse a full manifest parse above this size

Sampling is spread breadth-first across tasks before depth within a task, so
an axis frozen in one task but swept in another is caught. Raise
`--iters-per-campaign` if a campaign has many tasks and you want tighter
coverage; the cost is one small decompression per file.


## Reading the axis matrix (section 3 of the report)

    SWEPT [lo, hi]        declared swept, read by the simulator, column varies
    SWEPT(INERT)          drawn and recorded, read by nothing
    SWEPT(const!)         declared swept, but the sampled column never varied
    VARIES(undeclared)    column varies although the declaration omits it
    fixed=V               held at V
    -                     axis absent from that campaign's registry

`SWEPT(INERT)` and `VARIES(undeclared)` are the two findings that a plain
variance scan over the exported `(theta, x)` pairs cannot produce.


## Known limits

  * The empirical scan is a SAMPLE. An axis reported constant is constant
    across the sampled rows, not proven constant across every row. Raise
    `--iters-per-campaign` when that distinction matters.
  * Inertness for pre-`axis_declaration` campaigns is INFERRED from `mode`
    only. If a campaign had an axis inert for some other reason, the tool
    will not know; the report states which campaigns are in this position.
  * Independent sample size along topology-level axes (`conn_prob`, or the
    weibull `(p0, d0, beta)` triple) is the number of TOPOLOGIES, not the
    number of rows -- every row within a topology shares the same draw
    exactly. The observed-range columns in section 7 report distinct values,
    which makes this visible, but the tool does not correct for it.
