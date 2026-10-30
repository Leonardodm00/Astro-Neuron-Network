import json, os, sys, tempfile, shutil
here = sys.argv[1]
sys.path.insert(0, here)
import HPC_main_sweep as S
opts = {o for act in S.build_parser()._actions for o in act.option_strings}
gate = ['--stoa_gate_lo', '1.0', '--stoa_gate_hi', '1.0'] if '--stoa_gate_lo' in opts else []
tmp = tempfile.mkdtemp(prefix='ja_')
fr = 'Cm,I_inj,VA=-50,DeltaA,VR=-55,EC50_ampa,EC50_nmda,alpha_syn,O_G,Omega_G'
sys.argv = ['HPC_main_sweep.py', '--out_dir', tmp, '--lib_dir', here, '--n_topologies', '0',
            '--n_workers', '1', '--mode', 'Neuronal', '--sweep_group', 'neuron_synapse',
            '--freeze', fr, '--density', '1300', '--c_max', '350', '--Na', '0',
            '--simtime', '180', '--seed_master', '12345'] + gate
S.main()
d = json.load(open(os.path.join(tmp, 'job_args.json')))
print('N_DIMS', S.N_DIMS)
print('keys present:', {k: (k in d) for k in ('_active_param_names', 'freeze', 'Nn', 'Na', 'mode', 'simtime')})
print('free', len(d['_active_param_names']), d['_active_param_names'])
print('freeze', d['freeze']); print('Nn', d['Nn'], 'Na', d['Na'], 'mode', d['mode'], 'simtime', d['simtime'])
fx = d['_axis_declaration']['fixed_axes']
for n in ('Cm', 'I_inj', 'VA', 'DeltaA', 'VR', 'EC50_ampa', 'EC50_nmda', 'alpha_syn', 'O_G', 'Omega_G', 'DeltaT', 'VT', 'gL'):
    k = S.PARAM_NAMES.index(n)
    lo, hi = S.PARAM_BOUNDS[k]
    print('%-10s fixed %-10s %-8s box [%g, %g] %s' % (n, fx.get(n), S.PARAM_UNITS[k], lo, hi, 'LOG' if k in S.LOG_PARAMS else ''))
for n in d['_active_param_names']:
    k = S.PARAM_NAMES.index(n); lo, hi = S.PARAM_BOUNDS[k]
    print('FREE %-10s nominal %-10g %-8s box [%g, %g] %s' % (n, S.NOMINAL_PARAMS[k], S.PARAM_UNITS[k], lo, hi, 'LOG' if k in S.LOG_PARAMS else ''))
shutil.rmtree(tmp)
