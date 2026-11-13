"""Helper for test_spec_block06_range_check.py (not a test): in a fresh process,
import range_check.py and a setup's HPC_main_sweep.py, build one level's
design with build_level() (what --rc-design-only prints, without running
anything) and dump it as JSON.

usage: python _rc_design_probe.py RANGE_DIR LIB_DIR PHASE LEVEL OUT_JSON -- <sweep settings>
"""
import json
import os
import sys

sys.dont_write_bytecode = True


def main():
    range_dir, lib_dir, phase, level, out_json = sys.argv[1:6]
    rest = sys.argv[sys.argv.index('--') + 1:]
    sys.path.insert(0, range_dir)
    import range_check as RC
    rc, other = RC.rc_parser().parse_known_args(['--rc-phase', phase, '--rc-level', level,
                                                 '--rc-out', '/nonexistent'] + rest)
    S = RC.load_sweep_module(lib_dir)
    parser = S.build_parser()
    args = parser.parse_args(other + ['--lib_dir', lib_dir, '--out_dir', 'x',
                                      '--n_params_per_worker', '1'])
    n_workers = RC.resolve_n_workers(S, args)
    design, padded, conn = RC.build_level(S, args, rc, n_workers)
    out = dict(design=design, padded=[list(map(float, p)) for p in padded], conn=conn,
               param_names=list(S.PARAM_NAMES), log_params=sorted(int(k) for k in S.LOG_PARAMS),
               bounds=[list(map(float, b)) for b in S.PARAM_BOUNDS],
               bounds_theta=[list(map(float, b)) for b in S.PARAM_BOUNDS_THETA],
               nominal=list(map(float, S.NOMINAL_PARAMS)), n_workers=n_workers)
    with open(out_json, 'w') as fh:
        json.dump(out, fh)


if __name__ == '__main__':
    main()
