"""Check: with the code's noise term sigma*sqrt(2*gl_ref/Cm)*randn()/sqrt(dt) held constant
over each step and exponential-Euler integration of the leak, the stationary SD of V of the
passive membrane (no g_A, no synaptic input, no spike term, gl = gl_ref) equals sigma.
Truth chosen: sigma = 4 mV, gl = gl_ref = 1.15 nS, Cm = 17.25 pF (tau_m = 15 ms), dt = 0.05 ms,
seed 1; tolerance: |SD/sigma - 1| < 0.02 over 2,000 independent cells x 2 s after a 0.3 s burn-in."""
import numpy as np
CFG = dict(sigma=4e-3, gl=1.15e-9, Cm=17.25e-12, dt=0.05e-3, T=2.0, burn=0.3, n=2000, seed=1)
rng = np.random.default_rng(CFG['seed'])
tau = CFG['Cm'] / CFG['gl']; a = np.exp(-CFG['dt'] / tau)
V = np.zeros(CFG['n'])                       # V - E_L
nb, nt = int(CFG['burn'] / CFG['dt']), int(CFG['T'] / CFG['dt'])
acc, acc2, m = 0.0, 0.0, 0
for k in range(nb + nt):
    noise = CFG['sigma'] * np.sqrt(2 * CFG['gl'] / CFG['Cm']) * rng.standard_normal(CFG['n']) / np.sqrt(CFG['dt'])
    V = V * a + tau * (1 - a) * noise        # exponential Euler, B = noise held over dt
    if k >= nb and k % 20 == 0:
        acc += V.sum(); acc2 += (V**2).sum(); m += V.size
sd = np.sqrt(acc2 / m - (acc / m) ** 2)
print('tau_m = %.1f ms, SD(V) = %.3f mV vs sigma = %.3f mV, ratio %.4f -> %s'
      % (tau * 1e3, sd * 1e3, CFG['sigma'] * 1e3, sd / CFG['sigma'], 'PASS' if abs(sd / CFG['sigma'] - 1) < 0.02 else 'FAIL'))
