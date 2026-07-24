#!/usr/bin/env python3
# build_mea_manifest.py
# =============================================================================
# Discover every sweep_*_task* directory (one MEA-analysis work unit each)
# under one or more campaign roots, and write a numbered manifest that
# submit_mea_array.sh indexes into via $PBS_ARRAY_INDEX.
#
# A "work unit" is a directory that directly contains topo_*/ subdirectories
# -- i.e. exactly what --campaign expects in process_campaign.py. This mirrors
# HPC_main_sweep.py's own output layout:
#   <campaign_root>/sweep_<NODETAG>_task<IDX>/topo_<k>/...
#
# WHY A SEPARATE DISCOVERY STEP
#   PBS array jobs need to know the task COUNT at submission time (the -J
#   0-N range), and each array member needs a way to know WHICH directory is
#   "theirs" without re-globbing the filesystem (races, and inconsistent
#   ordering across nodes/runs). A manifest file, built once up front and
#   read by line number, is the standard fix for both problems.
#
# OUTPUT PATH CONSTRUCTION
#   To avoid collisions when the same "sweep_cpu_task0000"-style name repeats
#   across different campaigns, the output path preserves the last TWO path
#   components (the campaign directory name and the sweep/task directory
#   name) under --out-root:
#       <out-root>/<campaign_dir_name>/<sweep_task_dir_name>
#
# USAGE
#   python3 build_mea_manifest.py \
#       --campaign-root /path/campaign_A --campaign-root /path/campaign_B \
#       --out-root /path/mea_out \
#       --manifest manifest.tsv
#   # or, to sweep every campaign_* directory under a parent in one go:
#   python3 build_mea_manifest.py \
#       --glob '/path/to/Main/campaign_*' --out-root /path/mea_out
#
#   python3 build_mea_manifest.py --help
# =============================================================================

import argparse
import glob
import os
import sys


def find_work_units(campaign_root):
    """
    Every immediate subdirectory of campaign_root that itself directly
    contains at least one topo_* directory. Returns absolute paths, sorted.
    """
    campaign_root = os.path.abspath(campaign_root)
    if not os.path.isdir(campaign_root):
        print('WARNING: not a directory, skipping: %s' % campaign_root,
              file=sys.stderr)
        return []
    units = []
    for entry in sorted(os.listdir(campaign_root)):
        cand = os.path.join(campaign_root, entry)
        if not os.path.isdir(cand):
            continue
        has_topo = any(
            name.startswith('topo_') and
            os.path.isdir(os.path.join(cand, name))
            for name in os.listdir(cand))
        if has_topo:
            units.append(cand)
    return units


def out_path_for(campaign_dir, out_root):
    """<out-root>/<campaign_dir parent name>/<campaign_dir name>."""
    campaign_dir = os.path.abspath(campaign_dir)
    sweep_name = os.path.basename(campaign_dir)
    campaign_name = os.path.basename(os.path.dirname(campaign_dir))
    return os.path.join(os.path.abspath(out_root), campaign_name, sweep_name)


def is_already_done(out_dir, min_iters=1):
    """
    True if out_dir looks like a completed run: it has an mea_manifest.json
    (written by process_campaign.walk_campaign's _write_manifest) reporting
    at least min_iters processed. Best-effort only -- a partially completed
    run will not be recognised as done, which is the safe direction to be
    wrong in.
    """
    man = os.path.join(out_dir, 'mea_manifest.json')
    if not os.path.isfile(man):
        return False
    try:
        import json
        with open(man) as f:
            d = json.load(f)
        return int(d.get('total_done', 0)) >= min_iters
    except Exception:  # noqa: BLE001 -- any parse issue => treat as not done
        return False


def build_parser():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--campaign-root', action='append', default=[],
                   help='a campaign_<TAG> directory (repeatable)')
    p.add_argument('--glob', action='append', default=[],
                   help="shell glob matching multiple campaign_<TAG> dirs, "
                        "e.g. '/path/to/Main/campaign_*' (quote it so your "
                        "shell does not expand it early; repeatable)")
    p.add_argument('--out-root', required=True,
                   help='root output directory; per-unit dirs are created '
                        'under here (see module docstring for the layout)')
    p.add_argument('--manifest', default='mea_manifest.tsv',
                   help='path to write the manifest (default: ./mea_manifest.tsv)')
    p.add_argument('--skip-done', action='store_true',
                   help='omit work units whose out-dir already has a '
                        'manifest.json reporting completed iterations')
    return p


def main():
    args = build_parser().parse_args()

    roots = list(args.campaign_root)
    for pattern in args.glob:
        matches = sorted(glob.glob(pattern))
        if not matches:
            print('WARNING: glob matched nothing: %s' % pattern, file=sys.stderr)
        roots.extend(matches)

    if not roots:
        raise SystemExit('no --campaign-root or --glob given; nothing to do')

    rows = []
    skipped_done = 0
    for root in roots:
        units = find_work_units(root)
        if not units:
            print('WARNING: no sweep_*_task* (with topo_*) found under: %s'
                  % root, file=sys.stderr)
        for u in units:
            od = out_path_for(u, args.out_root)
            if args.skip_done and is_already_done(od):
                skipped_done += 1
                continue
            rows.append((u, od))

    if not rows:
        raise SystemExit('found 0 work units to process (after --skip-done '
                         'filtering)' if skipped_done else
                         'found 0 work units under the given roots')

    with open(args.manifest, 'w') as f:
        for campaign_dir, out_dir in rows:
            f.write('%s\t%s\n' % (campaign_dir, out_dir))

    print('[manifest] %d work unit(s) written -> %s'
          % (len(rows), os.path.abspath(args.manifest)))
    if skipped_done:
        print('[manifest] %d already-done unit(s) skipped (--skip-done)'
              % skipped_done)
    print('[manifest] array range for qsub: -J "0-%d"' % (len(rows) - 1))


if __name__ == '__main__':
    main()
