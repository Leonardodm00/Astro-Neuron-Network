"""Helper for test_spec_block10_median_drive.py (not a test): build the library's
Synapse_to_astro on stand-in groups with given Y_A values and links, run a few
steps on the requested device, and save Y_extra per step with the inputs.

usage: python _median_probe.py PIPE DEVICE(runtime|cpp) IN_NPZ OUT_NPZ [BUILD_DIR]
IN_NPZ: y_A (n_syn,) in mM, src (n_links,), tgt (n_links,), n_astro, tau_ms (0 = constant Y_A)
"""
import sys

import numpy as np

sys.dont_write_bytecode = True


def main():
    pipe, device, inp, out = sys.argv[1:5]
    sys.path.insert(0, pipe)
    import brian2 as b2
    import ASD_fun_BD_cpp as L
    if device == 'cpp':
        b2.set_device('cpp_standalone', build_on_run=False, directory=sys.argv[5])
    else:
        b2.set_device('runtime')
        b2.prefs.codegen.target = 'numpy'
    d = np.load(inp)
    b2.defaultclock.dt = 0.05 * b2.ms
    tau = float(d['tau_ms'])
    if tau > 0:
        syn = b2.NeuronGroup(len(d['y_A']), 'dY_A/dt = -Y_A/tau_y : mole', namespace=dict(tau_y=tau * b2.ms),
                             name='Synapse', dtype=np.float32 if False else None)
    else:
        syn = b2.NeuronGroup(len(d['y_A']), 'Y_A : mole', namespace={}, name='Synapse')
    syn.Y_A = d['y_A'] * b2.mmole
    astro = b2.NeuronGroup(int(d['n_astro']), 'Y_extra : mole', namespace={}, name='Astrocyte')
    links, pairs = L.Synapse_to_astro(syn, astro, connections=[d['src'], d['tgt']])
    m_y = b2.StateMonitor(syn, 'Y_A', record=True, when='end', name='m_y')
    m_e = b2.StateMonitor(astro, 'Y_extra', record=True, when='end', name='m_e')
    net = b2.Network(syn, astro, links, m_y, m_e)
    net.run(5 * b2.defaultclock.dt)
    if device == 'cpp':
        b2.device.build(run=True, directory=sys.argv[5])
    np.savez(out, y_end=np.asarray(m_y.Y_A / b2.mmole), extra_end=np.asarray(m_e.Y_extra / b2.mmole),
             pairs0=np.asarray(pairs[0]), pairs1=np.asarray(pairs[1]),
             has_model=len(links.equations) if hasattr(links, 'equations') else -1)


if __name__ == '__main__':
    main()
