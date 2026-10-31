#!/usr/bin/env python3
# =============================================================================
# env_check.py -- verify a conda environment can actually run this pipeline.
#
# Written after brian_env was found with a MISSING STDLIB (lib/python3.11/urllib/
# absent entirely), which surfaced only as an obscure ModuleNotFoundError deep
# inside `from pathlib import Path`. That failure mode is why this script tests
# the standard library explicitly instead of assuming it: a broken interpreter
# does not announce itself, it just fails somewhere unrelated later.
#
# Run it from the MEAMA folder, in the environment under test. Use the
# ABSOLUTE PATH to the env's python, not bare `python` -- on this cluster,
# `conda activate` can succeed (CONDA_PREFIX correctly set) while bare
# `python` on PATH still resolves to a module-loaded system interpreter
# instead (confirmed 2026-08-31: /archive/apps/Python/3.12.13/bin/python won
# over the activated env's own 3.11). Check [1] below catches this if you
# invoke it wrong, but invoking it right avoids the false run entirely:
#     conda activate brian_final
#     cd /davinci-1/home/ldellamea/ANN/Phenomenological/MEAMA
#     "$CONDA_PREFIX/bin/python" env_check.py
#
# Exit status 0 iff every check passes. The cpp_standalone test is the one that
# matters most -- it is the only check that exercises the C++ toolchain, the
# code generator and the device layer together, which is exactly what every
# simulation does and what a fresh environment most often gets wrong.
# =============================================================================

import importlib
import os
import shutil
import subprocess
import sys
import tempfile

_pass = _fail = 0
_failed_names = []


def check(cond, msg, detail=''):
    global _pass, _fail
    if cond:
        _pass += 1
        print(f'  ok   {msg}' + (f'  [{detail}]' if detail else ''))
    else:
        _fail += 1
        _failed_names.append(msg)
        print(f'  FAIL {msg}' + (f'  -- {detail}' if detail else ''))


def section(t):
    print(f'\n{t}\n' + '-' * len(t))


# -----------------------------------------------------------------------------
def t_interpreter():
    section('[1] interpreter identity')
    print(f'  sys.executable : {sys.executable}')
    print(f'  version        : {sys.version.split()[0]}')
    print(f'  CONDA_PREFIX   : {os.environ.get("CONDA_PREFIX", "<unset>")}')
    prefix = os.environ.get('CONDA_PREFIX', '')
    check(bool(prefix), 'CONDA_PREFIX is set (an environment is active)')
    if prefix:
        ok = sys.executable.startswith(prefix)
        check(ok, 'the running python belongs to the ACTIVE environment',
              f'if this fails, `python` is resolving to another install '
              f'(module-loaded system python) and every check below is '
              f'testing the wrong interpreter')
        if not ok:
            print(f'\n  Re-run this exact command instead:\n'
                  f'      {prefix}/bin/python {os.path.abspath(__file__)}\n'
                  f'  Every script in this folder (bench_sizing.sh, '
                  f'submit_MEAMA.sh, make_brian_final.sh) already resolves '
                  f'the interpreter this same way -- via CONDA_PREFIX, never '
                  f'bare `python` -- so they are not affected by this even '
                  f'though this interactive check just was.')


def t_stdlib():
    section('[2] standard library (the brian_env failure mode)')
    # urllib first: its absence is what broke brian_env. pathlib imports it,
    # and pathlib is imported by essentially every script in the pipeline.
    for m in ('urllib', 'urllib.parse', 'pathlib', 'json', 'sqlite3', 'ssl',
              'lzma', 'bz2', 'zlib', 'ctypes', 'multiprocessing', 'email',
              'http', 'xml', 'csv', 'hashlib', 'socket', 'select'):
        try:
            importlib.import_module(m)
            check(True, f'import {m}')
        except Exception as e:
            check(False, f'import {m}', f'{type(e).__name__}: {e}')

    # Count stdlib entries; a gutted install shows up immediately.
    libdir = os.path.dirname(os.__file__)
    try:
        n = len(os.listdir(libdir))
        check(n > 150, f'stdlib directory is populated ({n} entries)',
              f'{libdir}; a healthy 3.11 has 200+')
    except OSError as e:
        check(False, 'stdlib directory readable', str(e))


def t_thirdparty():
    section('[3] third-party packages required by the pipeline')
    # Derived from the actual imports in HPC_main_sweep.py, HPC_single_run.py,
    # ASD_fun_BD_cpp.py, aggregate_sweep.py, burst_metrics.py, burst_plots.py,
    # seed_alloc.py and the smoke tests.
    for mod, pip_name in [('numpy', 'numpy'), ('scipy', 'scipy'),
                          ('pandas', 'pandas'), ('matplotlib', 'matplotlib'),
                          ('mpl_toolkits', 'matplotlib'), ('seaborn', 'seaborn'),
                          ('sklearn', 'scikit-learn'), ('brian2', 'brian2')]:
        try:
            m = importlib.import_module(mod)
            check(True, f'import {mod}', getattr(m, '__version__', 'no __version__'))
        except Exception as e:
            check(False, f'import {mod}  (install: {pip_name})',
                  f'{type(e).__name__}: {e}')


def t_version_compat():
    section('[3b] known incompatible version pairings')
    # NumPy 2.0 removed the ndarray.ptp METHOD (np.ptp the function survives).
    # brian2 <= 2.9 calls np.ndarray.ptp at import time and dies with
    #   AttributeError: type object 'numpy.ndarray' has no attribute 'ptp'
    # Encountered for real on 2026-08-31 (brian2 2.9.0 + numpy 2.x). Checked
    # explicitly so the failure reads as a version problem rather than as an
    # obscure AttributeError from deep inside brian2's unit system.
    try:
        import numpy as np
    except Exception:
        check(False, 'numpy importable (prerequisite for this check)')
        return

    np_major = int(np.__version__.split('.')[0])
    has_ptp_method = hasattr(np.ndarray, 'ptp')

    try:
        import brian2
        b2v = brian2.__version__
        parts = b2v.split('.')
        b2_tuple = (int(parts[0]), int(parts[1]) if len(parts) > 1 else 0)
    except Exception as e:
        # brian2 import failure is already reported in [3]; explain it here if
        # it matches the known-bad signature.
        if np_major >= 2 and not has_ptp_method and 'ptp' in str(e):
            check(False,
                  'brian2 <= 2.9 is INCOMPATIBLE with numpy 2.x',
                  f'numpy {np.__version__} removed ndarray.ptp. Fix: '
                  f'pip install -U "brian2>=2.10"  (or pin "numpy<2")')
        else:
            check(False, 'brian2 importable (prerequisite for this check)',
                  f'{type(e).__name__}: {str(e)[:200]}')
        return

    if np_major >= 2:
        check(b2_tuple >= (2, 10),
              f'brian2 {b2v} is numpy-2 compatible (needs >= 2.10)',
              f'numpy {np.__version__} removed ndarray.ptp; brian2 <= 2.9 '
              f'calls it at import. Fix: pip install -U "brian2>=2.10"')
    else:
        check(True, f'numpy {np.__version__} is 1.x; the ndarray.ptp '
                    f'incompatibility does not apply', f'brian2 {b2v}')


def t_matplotlib_headless():
    section('[4] matplotlib on a headless node')
    try:
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt
        fig = plt.figure()
        fig.add_subplot(111).plot([0, 1], [0, 1])
        fd, png = tempfile.mkstemp(suffix='.png')
        os.close(fd)
        fig.savefig(png, dpi=50)
        plt.close(fig)
        ok = os.path.getsize(png) > 0
        os.unlink(png)
        check(ok, 'Agg backend renders and writes a PNG',
              'compute nodes have no display; the pipeline writes '
              'spatial_layout.png per topology')
    except Exception as e:
        check(False, 'matplotlib Agg render', f'{type(e).__name__}: {e}')


def t_toolchain():
    section('[5] C++ toolchain (cpp_standalone needs it at run time)')
    for exe in ('gcc', 'g++', 'make'):
        p = shutil.which(exe)
        check(p is not None, f'{exe} on PATH', p or 'NOT FOUND')
    gpp = shutil.which('g++')
    if gpp:
        try:
            v = subprocess.run([gpp, '--version'], capture_output=True,
                               text=True, timeout=30).stdout.splitlines()[0]
            check(True, 'g++ reports a version', v)
        except Exception as e:
            check(False, 'g++ --version', str(e))


def t_brian_standalone():
    section('[6] brian2 cpp_standalone: COMPILE AND RUN a real network')
    # This is the decisive test. It is the only one that exercises the code
    # generator, the C++ compiler and the device layer together -- i.e. what
    # every simulation in the campaign does. An environment can import brian2
    # cleanly and still fail here.
    tmp = tempfile.mkdtemp(prefix='brian_envcheck_')
    try:
        import brian2 as b2
        b2.set_device('cpp_standalone', build_on_run=False, directory=tmp)
        b2.prefs.codegen.cpp.extra_compile_args = ['-w', '-O1']
        tau = 10 * b2.ms          # namespace constant: Brian equation strings
        G = b2.NeuronGroup(       # cannot contain attribute access like b2.ms
            4, 'dv/dt = (1.1 - v) / tau : 1',
            threshold='v > 1', reset='v = 0', method='exact',
            namespace={'tau': tau})
        G.v = 0
        mon = b2.SpikeMonitor(G)
        net = b2.Network(G, mon)
        net.run(50 * b2.ms)
        b2.device.build(directory=tmp, compile=True, run=True, debug=False)
        n = int(mon.num_spikes)
        check(n > 0, 'cpp_standalone compiled, linked, ran, and produced spikes',
              f'{n} spikes from 4 neurons in 50 ms')
    except Exception as e:
        check(False, 'cpp_standalone build/run',
              f'{type(e).__name__}: {str(e)[:400]}')
    finally:
        try:
            b2.device.reinit()
        except Exception:
            pass
        shutil.rmtree(tmp, ignore_errors=True)


def t_pipeline():
    section('[7] the pipeline itself (run this from the MEAMA folder)')
    here = os.path.dirname(os.path.abspath(__file__))
    if here not in sys.path:
        sys.path.insert(0, here)

    for f in ('HPC_main_sweep.py', 'HPC_single_run.py', 'ASD_fun_BD_cpp.py',
              'aggregate_sweep.py', 'burst_metrics.py', 'burst_plots.py',
              'seed_alloc.py', 'synapse_pdist.csv'):
        check(os.path.isfile(os.path.join(here, f)), f'{f} present in {here}')

    try:
        S = importlib.import_module('HPC_main_sweep')
        check(S.N_DIMS == 37, f'registry is 37-D (got {S.N_DIMS})',
              'if 36, this copy predates the O_N port')
        n = len(S.resolve_sweep_group('tripartite'))
        check(n == 32, f"sweep group 'tripartite' has 32 axes (got {n})")
        check(hasattr(S, 'stoa_gate_record_keys'),
              'astrocyte contact gate present (stoa_gate_record_keys)')
    except Exception as e:
        check(False, 'import HPC_main_sweep', f'{type(e).__name__}: {e}')

    try:
        import pandas as pd
        df = pd.read_csv(os.path.join(here, 'synapse_pdist.csv'))
        ok = {'Syn_prob', 'Radius_val'} <= set(df.columns)
        check(ok, 'synapse_pdist.csv has Syn_prob and Radius_val',
              f'{len(df)} rows')
    except Exception as e:
        check(False, 'read synapse_pdist.csv', f'{type(e).__name__}: {e}')


def main():
    print('=' * 72)
    print('env_check -- can this environment run the MEAMA pipeline?')
    print('=' * 72)
    t_interpreter()
    t_stdlib()
    t_thirdparty()
    t_version_compat()
    t_matplotlib_headless()
    t_toolchain()
    t_brian_standalone()
    t_pipeline()
    print('\n' + '=' * 72)
    print(f'RESULT: {_pass} passed, {_fail} failed')
    if _fail:
        print('\nfailed checks:')
        for n in _failed_names:
            print(f'  - {n}')
    print('=' * 72)
    return 1 if _fail else 0


if __name__ == '__main__':
    sys.exit(main())
