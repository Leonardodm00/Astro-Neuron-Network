"""Shared helpers for the spec-derived tests (independent integration tester).

The tests read specs/SPEC.md's contracts; expected values are computed here
from the spec's formulas with NumPy/SciPy, never by calling the code under
test. Nothing is written inside the repository: every file a test needs is
created under pytest's tmp_path (byte-code writing is switched off too).
"""
import os
import shutil
import subprocess
import sys

sys.dont_write_bytecode = True
os.environ.setdefault('PYTHONDONTWRITEBYTECODE', '1')

import pytest

REPO = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
HPC = os.path.join(REPO, 'hpc')
PIPE = os.path.join(HPC, 'Phenomenological_finalv1')
AUDIT = os.path.join(HPC, 'refractory_audit')
RANGE = os.path.join(HPC, 'range_check')
LCDUP = os.path.join(PIPE, 'LindaColloDUP')
GG = os.path.join(PIPE, 'GiuliaGate')
PIPELINE_FILES = ('HPC_main_sweep.py', 'HPC_single_run.py', 'ASD_fun_BD_cpp.py',
                  'aggregate_sweep.py', 'burst_metrics.py', 'burst_plots.py',
                  'synapse_pdist.csv')


def run(cmd, cwd=None, env=None, timeout=1800, input=None):
    """subprocess.run with captured text output; never raises on exit status."""
    e = dict(os.environ)
    e['PYTHONDONTWRITEBYTECODE'] = '1'
    e['MPLBACKEND'] = 'Agg'
    if env:
        e.update(env)
    return subprocess.run(cmd, cwd=cwd, env=e, stdout=subprocess.PIPE, stderr=subprocess.PIPE,
                          universal_newlines=True, timeout=timeout, input=input)


def copy_tree(src, dst, ignore_names=('__pycache__',)):
    shutil.copytree(src, dst, ignore=shutil.ignore_patterns(*ignore_names, '*.pyc'))
    return dst


def package_with_pipeline(pkg_dir, tmp_dir):
    """A fresh copy of a launch package folder with the 7 root pipeline files
    copied in (step 0 of APPLY_LCDUP.txt / APPLY_GG.txt), under tmp_dir."""
    dst = os.path.join(str(tmp_dir), os.path.basename(pkg_dir))
    copy_tree(pkg_dir, dst)
    for f in PIPELINE_FILES:
        shutil.copy2(os.path.join(PIPE, f), os.path.join(dst, f))
    return dst


def import_from(folder, name):
    """Import module `name` from `folder` in a fresh module object."""
    import importlib.util
    spec = importlib.util.spec_from_file_location(name + '_under_test_%d' % abs(hash(folder)),
                                                  os.path.join(folder, name + '.py'))
    mod = importlib.util.module_from_spec(spec)
    sys.path.insert(0, folder)
    try:
        spec.loader.exec_module(mod)
    finally:
        try:
            sys.path.remove(folder)
        except ValueError:
            pass
    return mod


@pytest.fixture(scope='session')
def pipe_path():
    """The root pipeline folder on sys.path (import only; no file is written)."""
    if PIPE not in sys.path:
        sys.path.insert(0, PIPE)
    return PIPE


LCDUP_FREEZE = 'Cm,I_inj,VA=-48,DeltaA,VR=-58.2,EC50_ampa,EC50_nmda,alpha_syn,O_G,Omega_G'


@pytest.fixture(scope='session')
def lcdup_folder(tmp_path_factory):
    """LindaColloDUP folder after APPLY_LCDUP.txt steps 0 and 4 (both patches), in tmp."""
    d = package_with_pipeline(LCDUP, tmp_path_factory.mktemp('lcdup'))
    for script in ('apply_freeze_patch.py', 'apply_box_patch.py'):
        r = run([sys.executable, script], cwd=d)
        assert r.returncode == 0, r.stdout + r.stderr
    return d


@pytest.fixture(scope='session')
def gg_folder(tmp_path_factory):
    """GiuliaGate folder after APPLY_GG.txt steps 0 and 4 (freeze and prior patches), in tmp."""
    d = package_with_pipeline(GG, tmp_path_factory.mktemp('gg'))
    for script in ('apply_freeze_patch.py', 'apply_prior_patch.py'):
        r = run([sys.executable, script], cwd=d)
        assert r.returncode == 0, r.stdout + r.stderr
    return d


def brian2_runtime():
    """Switch Brian2 to the runtime device with the numpy target (no compiler)."""
    import brian2
    brian2.set_device('runtime')
    brian2.prefs.codegen.target = 'numpy'
    brian2.start_scope()
    return brian2
