#!/usr/bin/env python3
"""
check_reset_runaway_input.py -- standalone check (not pipeline code).

Claims tested
    (1) A constant net depolarizing current I lowers the unstable fixed point
        V_u(I) of  Cm dV/dt = gL (EL - V) + gL DT exp((V - VT)/DT) + I,
        and above the rheobase I_rh = gL (VT - EL - DT) no fixed point is
        left: V then runs away from any start, V_R included.
    (2) The smallest constant I that takes V from V_R to +inf within
        t_ref = 3 ms when no threshold can fire (as during t_ref in the
        sweep's code, where dV/dt has no "(unless refractory)").

Truth chosen (fixed before running)
    Registry cell: Cm 17.25 pF, gL 1.15 nS, EL -58.2 mV, VT -51 mV, DT 2 mV;
    gA = 0 (no adaptation), no noise, I constant over the 3 ms.
    V_R = -55 mV (LindaColloDUP default) and -58.2 mV (registry).
    Units: pF, nS, mV, ms, pA (nS x mV = pA). Blow-up time integrated in
    w = exp(-(V - VT)/DT), where V -> +inf is w -> 0 at a finite time; a
    constant I enters exactly as EL -> EL + I/gL (same integrator as
    check_blowup_time.py). solve_ivp RK45, rtol 1e-10, event at w = 1e-12.
"""
import numpy as np
from scipy.integrate import solve_ivp
from scipy.optimize import brentq

CFG = dict(Cm=17.25, gL=1.15, EL=-58.2, VT=-51.0, DT=2.0, t_ref=3.0, w_stop=1e-12,
           I_test=(0.0, 2.0, 4.0, 5.9), V_R=(-55.0, -58.2), t_cap=1e6)


def i_v(V, I, c):
    """Net membrane current [pA] at voltage V [mV] with constant input I [pA]."""
    return c['gL'] * (c['EL'] - V) + c['gL'] * c['DT'] * np.exp((V - c['VT']) / c['DT']) + I


def v_unstable(I, c):
    """Upper (unstable) zero of i_v above VT, or nan when I exceeds the rheobase."""
    return brentq(i_v, c['VT'], c['VT'] + 30.0, args=(I, c)) if i_v(c['VT'], I, c) < 0 else np.nan


def t_blowup(V0, I, c):
    """Time [ms] for V to reach +inf from V0 with constant I and no threshold."""
    tau_m, EL_eff = c['Cm'] / c['gL'], c['EL'] + I / c['gL']

    def rhs(t, w):
        x = max(w[0], 1e-300)
        return [-(1.0 + (x / c['DT']) * (EL_eff - c['VT'] + c['DT'] * np.log(x))) / tau_m]

    ev = lambda t, w: w[0] - c['w_stop']
    ev.terminal, ev.direction = True, -1
    s = solve_ivp(rhs, (0.0, 500.0), [np.exp(-(V0 - c['VT']) / c['DT'])], events=ev,
                  method='RK45', rtol=1e-10, atol=1e-14)
    return s.t_events[0][0] if s.t_events[0].size else c['t_cap']


def main():
    c = CFG
    I_rh = c['gL'] * (c['VT'] - c['EL'] - c['DT'])
    print('rheobase I_rh = gL (VT - EL - DT) = %.3f pA; net current at VT with I = 0: %.3f pA'
          % (I_rh, i_v(c['VT'], 0.0, c)))
    for I in c['I_test'] + (I_rh + 0.01,):
        print('  I = %6.3f pA: V_u = %s' % (I, 'none (above rheobase)' if np.isnan(v_unstable(I, c))
                                              else '%.2f mV' % v_unstable(I, c)))
    for VR in c['V_R']:
        I_min = brentq(lambda I: t_blowup(VR, I, c) - c['t_ref'], I_rh + 1e-6, 1000.0, xtol=1e-6)
        print('V_R = %.1f mV: smallest constant I that reaches +inf within %.0f ms: %.1f pA '
              '(check: t_blowup at that I = %.4f ms; at 0.9 x I_min: %.2f ms)'
              % (VR, c['t_ref'], I_min, t_blowup(VR, I_min, c), t_blowup(VR, 0.9 * I_min, c)))


if __name__ == '__main__':
    main()
