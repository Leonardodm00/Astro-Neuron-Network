#!/usr/bin/env python3
"""
check_blowup_time.py -- standalone check (not pipeline code).

Claim tested
    Without a threshold check (as during t_ref in the sweep's model), the CAdEx
    voltage equation of ASD_fun_BD_cpp.py, started above its unstable fixed
    point V_u, reaches infinity in finite time, and for V0 = VT + 3 DeltaT that
    time is shorter than t_ref = 3 ms. The exponential term alone gives
    t* = tau_m exp(-(V0 - VT)/DeltaT), tau_m = Cm/gL.

Truth chosen (fixed before running)
    Cm dV/dt = gL (EL - V) + gL DeltaT exp((V - VT)/DeltaT), with the registry
    values Cm 17.25 pF, gL 1.15 nS, EL -58.2 mV, VT -51 mV, DeltaT 2 mV; no
    adaptation current, no synaptic input, no noise (each of those is set to
    zero; gA, I_syn and noise are not part of this check). Integrated in the
    variable w = exp(-(V - VT)/DeltaT), which turns V -> +inf into w -> 0 at a
    finite rate (dw/dt -> -1/tau_m), so the blow-up time is the zero crossing
    of w: dw/dt = -(1/tau_m) [1 + (w/DeltaT)(EL - VT + DeltaT ln w)].
    scipy solve_ivp (RK45, rtol 1e-10), event at w = 1e-12 (time left below
    1e-12 tau_m). Exp-only check of the method: dw/dt = -1/tau_m exactly.
"""
import numpy as np
from scipy.integrate import solve_ivp
from scipy.optimize import brentq

CFG = dict(Cm=17.25, gL=1.15, EL=-58.2, VT=-51.0, DT=2.0, t_ref=3.0, w_stop=1e-12)   # pF, nS, mV, ms


def rhs_w(t, w, c, leak=True):
    tau_m = c['Cm'] / c['gL']
    x = max(w[0], 1e-300)
    corr = (x / c['DT']) * (c['EL'] - c['VT'] + c['DT'] * np.log(x)) if leak else 0.0
    return [-(1.0 + corr) / tau_m]


def t_to_blowup(V0, c, leak=True):
    ev = lambda t, w, c, leak: w[0] - c['w_stop']
    ev.terminal, ev.direction = True, -1
    w0 = np.exp(-(V0 - c['VT']) / c['DT'])
    s = solve_ivp(rhs_w, (0.0, 200.0), [w0], args=(c, leak), events=ev, method='RK45',
                  rtol=1e-10, atol=1e-14)
    return s.t_events[0][0] if s.t_events[0].size else np.inf


def main():
    c = CFG
    tau_m = c['Cm'] / c['gL']
    Vu = brentq(lambda v: c['gL'] * (c['EL'] - v) + c['gL'] * c['DT'] * np.exp((v - c['VT']) / c['DT']),
                c['VT'], c['VT'] + 10)
    print('tau_m = %.2f ms; unstable fixed point V_u = %.2f mV (VT + %.2f mV)' % (tau_m, Vu, Vu - c['VT']))
    # method check: exp-only integration must reproduce t* = tau_m exp(-(V0-VT)/DT)
    t_num = t_to_blowup(-45.0, c, leak=False)
    t_ana = tau_m * np.exp(-3.0)
    print('method check, exp-only, V0 = -45 mV: numerical %.6f ms vs analytic %.6f ms (rel. diff %.1e)'
          % (t_num, t_ana, abs(t_num - t_ana) / t_ana))
    for V0 in (-47.0, -46.0, -45.0, -44.0, -43.0):
        t_full = t_to_blowup(V0, c)
        t_exp = tau_m * np.exp(-(V0 - c['VT']) / c['DT'])
        print('V0 = %6.1f mV (VT + %.1f DeltaT): blow-up after %.3f ms with the leak, %.3f ms exp-only; < t_ref: %s'
              % (V0, (V0 - c['VT']) / c['DT'], t_full, t_exp, t_full < c['t_ref']))
    V3 = brentq(lambda v: t_to_blowup(v, c) - c['t_ref'], Vu + 1e-3, c['VT'] + 6 * c['DT'])
    print('starting voltage that blows up exactly within t_ref = 3 ms: %.2f mV (VT + %.2f DeltaT)'
          % (V3, (V3 - c['VT']) / c['DT']))


if __name__ == '__main__':
    main()
