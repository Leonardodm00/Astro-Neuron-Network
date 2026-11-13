"""Block 10 -- the astrocyte's glutamate drive is the median over its links
(SPEC.md Block 10).

Oracle: numpy.median of the linked Y_A computed here (0 without a link), on
random link structures with odd and even counts, ties and outliers; under the
runtime device (NumPy implementation) and under cpp_standalone (the custom
C++ implementation); timing: Y_extra at step n uses the Y_A left by step n - 1.
Static: links without a model, the run_regularly name and slot, the Function's
units. Run:
    python -m pytest tests/spec/test_spec_block10_median_drive.py -q
"""
import os
import re
import sys

import numpy as np
import numpy.testing as npt
import pytest

from conftest import PIPE, run

PROBE = os.path.join(os.path.dirname(__file__), '_median_probe.py')
SRC = open(os.path.join(PIPE, 'ASD_fun_BD_cpp.py'), encoding='utf-8').read()


def random_links(seed, n_syn=120, n_astro=30):
    rng = np.random.default_rng(seed)
    y = np.round(rng.lognormal(0.0, 1.0, n_syn), 2)               # rounding makes ties
    y[rng.random(n_syn) < 0.05] = 100.0                           # outliers
    counts = rng.integers(0, 10, n_astro)                         # N_k = 0..9, odd and even
    counts[:3] = [0, 1, 2]
    src, tgt = [], []
    for a, c in enumerate(counts):
        syn = rng.choice(n_syn, size=int(c), replace=False)
        src += syn.tolist()
        tgt += [a] * int(c)
    return y, np.array(src, dtype=np.int32), np.array(tgt, dtype=np.int32), n_astro


def expected(y, src, tgt, n_astro):
    out = np.zeros(n_astro)
    for a in range(n_astro):
        m = tgt == a
        out[a] = np.median(y[src[m]]) if m.any() else 0.0
    return out


def probe(tmp_path, device, y, src, tgt, n_astro, tau_ms=0.0, tag='p'):
    inp, out = str(tmp_path / ('%s_in.npz' % tag)), str(tmp_path / ('%s_out.npz' % tag))
    np.savez(inp, y_A=y, src=src, tgt=tgt, n_astro=n_astro, tau_ms=tau_ms)
    cmd = [sys.executable, PROBE, PIPE, device, inp, out] + ([str(tmp_path / ('%s_build' % tag))]
                                                             if device == 'cpp' else [])
    r = run(cmd, cwd=str(tmp_path), timeout=1200)
    assert r.returncode == 0, r.stdout[-3000:] + r.stderr[-3000:]
    return np.load(out)


@pytest.mark.parametrize('device', ['runtime', 'cpp'])
def test_median_constant_inputs(tmp_path, device):
    y, src, tgt, n_astro = random_links(7)
    res = probe(tmp_path, device, y, src, tgt, n_astro, tag=device)
    want = expected(y, src, tgt, n_astro)
    # Y_A is stored as float64 here (stand-in group); rel. 1e-12 is round-off of the mM <-> mole scaling
    for step in range(res['extra_end'].shape[1]):
        npt.assert_allclose(res['extra_end'][:, step], want, rtol=1e-12, atol=0)
    # the returned pairs are (astrocyte, synapse), as Astro_to_Syn expects
    npt.assert_array_equal(res['pairs0'], tgt)
    npt.assert_array_equal(res['pairs1'], src)


@pytest.mark.parametrize('device', ['runtime', 'cpp'])
def test_median_timing_uses_previous_step(tmp_path, device):
    y, src, tgt, n_astro = random_links(11, n_syn=60, n_astro=12)
    res = probe(tmp_path, device, y, src, tgt, n_astro, tau_ms=1.0, tag=device + 't')
    y_end, e_end = res['y_end'], res['extra_end']               # (n_syn, steps), (n_astro, steps)
    npt.assert_allclose(e_end[:, 0], expected(y, src, tgt, n_astro), rtol=1e-12)
    for n in range(1, e_end.shape[1]):
        # Y_extra set at the start of step n from the Y_A left by step n-1; Y_A decays 5 % per step,
        # so the previous and the current step differ far beyond the 1e-12 tolerance
        npt.assert_allclose(e_end[:, n], expected(y_end[:, n - 1], src, tgt, n_astro), rtol=1e-12)
        assert not np.allclose(e_end[:, n], expected(y_end[:, n], src, tgt, n_astro), rtol=1e-6) \
            or np.all(e_end[:, n] == 0)


def test_static_structure():
    body = SRC[SRC.index('def Synapse_to_astro('):SRC.index('def Astro_to_Syn(')]
    assert re.search(r"Synapses\(synapse, Astro,\s*model=''", body)
    assert re.search(r"run_regularly\('Y_extra = astro_median_Y_A\(i, t\)', when='start',\s*"
                     r"name='Astrocyte_Y_extra_median'\)", body)
    assert 'Y_extra_post' not in SRC.replace('# ', '')            # no summed or mean drive left
    fn = SRC[SRC.index('def astro_median_function('):SRC.index('def Astro_to_Syn(')]
    assert 'arg_units=[1, second]' in fn and 'return_unit=mole' in fn and 'stateless=False' in fn
    assert "add_implementation('cpp'" in fn
