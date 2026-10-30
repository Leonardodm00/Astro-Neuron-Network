#!/usr/bin/env python3
"""
check_rs_reset_adaptation.py -- standalone check (not pipeline code).

Question
    Do the frozen V_A, Delta_A, V_R of the LindaColloDUP campaign give the
    regular-spiking (adapting) firing of the CAdEx "Adaptive spiking" cell of
    Gorski et al. 2021 (Table 1), and which choice of (V_A, V_R) keeps the
    reset below the minimum of the V-nullcline (Gorski sec. 5: bursting needs
    V_R ABOVE that minimum)?

Truth chosen (fixed before running)
    The CAdEx equations of ASD_fun_BD_cpp.py (main @ d0ced88) with I_syn = 0,
    no noise, a step current I_s, Brian2 runtime (numpy target),
    exponential Euler, dt = 0.05 ms, 1 s per step, V(0) = E_L,
    g_A(0) = g_A_inf(E_L).
    Model cell (hiPSC-grounded, registry values): C_m 17.25 pF, g_L 1.15 nS,
    E_L -58.2 mV, V_T -51 mV, Delta_T 2 mV, E_A -70 mV, V_D -40 mV,
    t_ref 3 ms; adaptation at the registry nominal: gbar_A 0.9 nS,
    dg_A 0.135 nS, tau_A 200 ms.
    Variants of (V_A, Delta_A, V_R) [mV]:
      registry   (-48, 5, -58.2)   current NOMINAL_PARAMS
      RS_abs     (-50, 5, -55.0)   Gorski Table 1 adaptive row, absolute values
      RS_VTanch  (-51, 5, -56.0)   same offsets from V_T as that row (V_A = V_T, V_R = V_T - 5)
    Reference: the Gorski adaptive-spiking cell itself (C 200 pF, g_L 10 nS,
    E_L -60, V_T -50, Delta_T 2 [not tabulated; canonical value], E_A -70,
    V_A -50, Delta_A 5, V_R -55, gbar_A 10 nS, dg_A 1 nS, tau_A 200 ms,
    t_ref 5 ms), at its own I_s = 200 pA and at 1.5x / 2x its rheobase.

Reported per run: rheobase (fixed-point disappearance, computed on a grid),
spike count in 1 s, first and last ISI, adaptation index
A = mean_i (ISI_{i+1} - ISI_i)/(ISI_{i+1} + ISI_i) (Gorski eq. 6.1; A > 0 =
decelerating = adapting), and V_min, the voltage of the minimum of the
V-nullcline g_A(V) = [g_L(E_L-V) + g_L Delta_T e^{(V-V_T)/Delta_T} + I_s]/(V-E_A).

Pass criteria (fixed before running): the Gorski reference shows A > 0 and
V_R < V_min (sanity of the metric); each variant is reported, not judged.

Usage:  python3 -I check_rs_reset_adaptation.py
"""
import numpy as np
import brian2 as b2

CFG = dict(dt_ms=0.05, T_s=1.0, drives=(1.5, 2.0), seed=0)

MODEL = dict(Cm=17.25, gL=1.15, EL=-58.2, VT=-51.0, DT=2.0, EA=-70.0, VD=-40.0,
             tref=3.0, gbarA=0.9, dgA=0.135, tauA=200.0)
VARIANTS = {
    'registry':  dict(VA=-48.0, DA=5.0, VR=-58.2),
    'RS_abs':    dict(VA=-50.0, DA=5.0, VR=-55.0),
    'RS_VTanch': dict(VA=-51.0, DA=5.0, VR=-56.0),
}
GORSKI_RS = dict(Cm=200.0, gL=10.0, EL=-60.0, VT=-50.0, DT=2.0, EA=-70.0,
                 VD=-40.0, tref=5.0, gbarA=10.0, dgA=1.0, tauA=200.0,
                 VA=-50.0, DA=5.0, VR=-55.0)


def gA_inf(p, V):
    return p['gbarA'] / (1.0 + np.exp((p['VA'] - V) / p['DA']))


def rheobase(p):
    """Smallest I_s [pA] with no fixed point: max over V of the current that
    balances leak + spike term + steady-state adaptation (nS*mV = pA)."""
    V = np.linspace(p['EL'] - 5.0, p['VD'], 200001)
    h = (p['gL'] * (V - p['EL']) - p['gL'] * p['DT'] * np.exp((V - p['VT']) / p['DT'])
         + gA_inf(p, V) * (V - p['EA']))
    return float(h.max())


def v_nullcline_min(p, I):
    V = np.linspace(p['EA'] + 0.5, p['VD'], 200001)
    g = (p['gL'] * (p['EL'] - V) + p['gL'] * p['DT'] * np.exp((V - p['VT']) / p['DT']) + I) / (V - p['EA'])
    return float(V[np.argmin(g)])


def simulate(cells):
    """cells: list of (label, params dict, I_s [pA]); one Brian2 group."""
    b2.prefs.codegen.target = 'numpy'
    b2.start_scope()
    b2.seed(CFG['seed'])
    b2.defaultclock.dt = CFG['dt_ms'] * b2.ms
    eqs = '''
    dV/dt  = (gl*(El-V) + gl*DeltaT*exp((V-VT)/DeltaT) + gA*(EA-V) + Is)/Cm : volt
    dgA/dt = (gbarA/(1 + exp((VA-V)/DeltaA)) - gA)/tauA : siemens
    Cm : farad
    gl : siemens
    El : volt
    VT : volt
    DeltaT : volt
    EA : volt
    VA : volt
    DeltaA : volt
    VR : volt
    VD : volt
    gbarA : siemens
    delta_gA : siemens
    tauA : second
    tref : second
    Is : amp
    '''
    # Same membrane equation as the production model: V is NOT clamped during
    # t_ref (the threshold is simply not tested while refractory).
    n = len(cells)
    G = b2.NeuronGroup(n, eqs, threshold='V > VD', reset='V = VR; gA += delta_gA',
                       refractory='tref', method='exponential_euler')
    for k, (_, p, I) in enumerate(cells):
        G.Cm[k] = p['Cm'] * b2.pF;   G.gl[k] = p['gL'] * b2.nS
        G.El[k] = p['EL'] * b2.mV;   G.VT[k] = p['VT'] * b2.mV
        G.DeltaT[k] = p['DT'] * b2.mV; G.EA[k] = p['EA'] * b2.mV
        G.VA[k] = p['VA'] * b2.mV;   G.DeltaA[k] = p['DA'] * b2.mV
        G.VR[k] = p['VR'] * b2.mV;   G.VD[k] = p['VD'] * b2.mV
        G.gbarA[k] = p['gbarA'] * b2.nS; G.delta_gA[k] = p['dgA'] * b2.nS
        G.tauA[k] = p['tauA'] * b2.ms; G.tref[k] = p['tref'] * b2.ms
        G.Is[k] = I * b2.pA
        G.V[k] = p['EL'] * b2.mV
        G.gA[k] = gA_inf(p, p['EL']) * b2.nS
    M = b2.SpikeMonitor(G)
    b2.run(CFG['T_s'] * b2.second)
    out = []
    for k, (label, p, I) in enumerate(cells):
        t = np.sort(np.asarray(M.t[M.i == k] / b2.ms))
        isi = np.diff(t)
        A = float(np.mean((isi[1:] - isi[:-1]) / (isi[1:] + isi[:-1]))) if len(isi) >= 2 else float('nan')
        out.append(dict(label=label, I=I, n=len(t),
                        isi1=float(isi[0]) if len(isi) else float('nan'),
                        isiN=float(isi[-1]) if len(isi) else float('nan'),
                        min_isi=float(isi.min()) if len(isi) else float('nan'),
                        A=A, Vmin=v_nullcline_min(p, I), VR=p['VR'], tref=p['tref']))
    return out


def main():
    cells = []
    for name, v in VARIANTS.items():
        p = dict(MODEL, **v)
        Irh = rheobase(p)
        for m in CFG['drives']:
            cells.append(('%s  %.1fx I_rh' % (name, m), p, m * Irh))
    Irh_g = rheobase(GORSKI_RS)
    cells.append(('Gorski RS  I_s=200', GORSKI_RS, 200.0))
    for m in CFG['drives']:
        cells.append(('Gorski RS  %.1fx I_rh' % m, GORSKI_RS, m * Irh_g))
    res = simulate(cells)
    print('rheobase [pA]: ' + ', '.join('%s %.2f' % (n, rheobase(dict(MODEL, **v)))
                                       for n, v in VARIANTS.items())
          + ', Gorski RS %.1f' % Irh_g)
    print('%-24s %8s %6s %9s %9s %8s %8s %8s %s' % ('cell', 'I [pA]', 'spikes', 'ISI1[ms]',
          'ISIn[ms]', 'A', 'V_R', 'V_min', 'V_R<V_min'))
    for r in res:
        print('%-24s %8.2f %6d %9.1f %9.1f %8.3f %8.1f %8.2f %s' % (
            r['label'], r['I'], r['n'], r['isi1'], r['isiN'], r['A'], r['VR'], r['Vmin'],
            'yes' if r['VR'] < r['Vmin'] else 'NO'))
    ref = [r for r in res if r['label'].startswith('Gorski')]
    ok = all(r['A'] > 0 and r['VR'] < r['Vmin'] for r in ref)
    print('reference sanity (Gorski RS: A > 0 and V_R < V_min at every drive):', 'PASS' if ok else 'FAIL')


if __name__ == '__main__':
    main()
