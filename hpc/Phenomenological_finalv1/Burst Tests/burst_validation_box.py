#!/usr/bin/env python3
# burst_validation_box.py
#
# Burst-validation parameter designs for the CAdEx neuron-astrocyte sweep.
#
# Two designs, both run over the weibull topology prior (the campaign default):
#
#   DESIGN A -- DECISIVE adaptation test.
#       Sweep ONLY the adaptation axes {delta_gA, tauA, gbarA} over the
#       burst-capable box from the CAdEx adaptation report (section 8.7,
#       widened upward because the network drive is a synaptic barrage, not the
#       single-cell constant step -- report section 8.4/9.4). EVERY other axis is
#       FROZEN at a burst-FAVOURABLE strong-ignition point (not registry nominal),
#       so any emergent burst<->pause regime is attributable to adaptation alone.
#       To keep the attribution clean while still sweeping topologies, run this as
#       a STRATIFIED sweep: reuse the SAME topology seeds at every adaptation grid
#       point (see the launch note at the bottom), so topology is a controlled
#       repeated factor and the adaptation effect is read WITHIN topology.
#
#   DESIGN B -- JOINT burst-permissive sweep.
#       Sweep the burst-relevant neuron+synapse axes, each NARROWED to its
#       burst-favourable sub-range, to map the burst region / build an SBI prior.
#       Astro axes and burst-irrelevant shaping axes are frozen.
#
# This module is a PURE transform of the registry: apply_design() takes
# (param_names, param_bounds, nominal) and returns the modified
# (active_idx, param_bounds, nominal). It imports nothing from the heavy driver,
# so it is trivially testable (smoke_test runs on numpy alone).
#
# Provenance of the numbers (all WITHIN the production PARAM_BOUNDS):
#   adaptation box   : CAdEx_Adaptation_Mapping_Report.md sections 6.1, 8.7
#   AMPAR-led ignition, NMDAR-set duration : Kleefstra/MEA (NBQX abolishes
#                      bursts; D-AP5 only shortens them)
#   U_0_sr depressing regime 0.2-0.6, tau_d 0.5-3 s : Kusick2020 / De Pitta;
#                      New_Synapse_Model_Report.md
#   Rates carry their reciprocal time constants explicitly:
#     Omega_d [1/s]    -> tau_d   = 1/Omega_d   (depression recovery)
#     Omega_f_sr [1/s] -> tau_f   = 1/Omega_f_sr(facilitation decay)
#
# ASCII-only source (HPC locale safe).

import numpy as np


# ===========================================================================
# Design definitions  (edit these freely; each value is independent)
# ===========================================================================
# 'sweep_bounds' : axis -> (lo, hi)  swept (must lie inside production bounds)
# 'freeze_values': axis -> value     frozen at a burst-favourable point
#                                     (overrides registry NOMINAL_PARAMS)
#
# Reciprocal-rate note kept explicit in the comments:
#   Omega_d = 0.6  /s  <=>  tau_d = 1/0.6   = 1.67 s   (inter-burst recovery)
#   Omega_f_sr = 2.5/s <=>  tau_f = 1/2.5   = 0.40 s   (Syt7 facilitation)

DESIGN_A = {
    'sweep_bounds': {
        'delta_gA': (0.4,   2.0),    # nS   report box, widened up for net drive
        'tauA':     (350.0, 1000.0), # ms   mandatory increase from nominal 200
        'gbarA':    (0.01,  0.5),    # nS   low subthreshold adaptation (-> paper 0)
    },
    'freeze_values': {
        # ignition (strong, AMPAR-led)
        'g_ampa':     1.0,    # nS
        'g_nmda':     0.4,    # nS
        'EC50_ampa':  7.0,    # mM
        'EC50_nmda':  3.0,    # mM
        'U_0_sr':     0.4,    # depressing regime -> deplete during burst
        # termination / recovery (synaptic, supports adaptation)
        'Omega_d':    0.6,    # 1/s  (tau_d = 1.67 s)
        'Omega_f_sr': 2.5,    # 1/s  (tau_f = 0.40 s)
        'U_0_ar':     1.0e-3, # keep async release low -> crisp inter-burst gaps
        # drive / excitability
        'I_inj':      4.5,    # pA
        'Cm':         17.25,  # pF   FIX tau_m=15 ms (clean adaptation test)
        'Sigma':      4.0,    # mV
    },
}

DESIGN_B = {
    'sweep_bounds': {
        # adaptation
        'delta_gA': (0.4,   2.0),
        'tauA':     (350.0, 1000.0),
        'gbarA':    (0.01,  0.5),
        # ignition
        'g_ampa':   (0.5,   2.0),
        'g_nmda':   (0.2,   0.8),
        'EC50_ampa':(3.0,   15.0),
        'EC50_nmda':(1.0,   8.0),
        'U_0_sr':   (0.2,   0.6),
        # termination / recovery + drive
        'Omega_d':   (0.3,  1.5),    # tau_d in [0.67, 3.33] s
        'Omega_f_sr':(1.5,  3.3),    # tau_f in [0.30, 0.67] s
        'I_inj':     (3.0,  6.0),
        'Cm':        (15.0, 30.0),
        'Sigma':     (2.0,  6.0),
    },
    'freeze_values': {
        'U_0_ar': 1.0e-3,            # keep async release low
        # everything else (U_max, alpha_syn, x0, Omega_f_ar, VA, DeltaA, VR,
        # and the astro axes / O_G / Omega_G which are inert in MODE=Neuronal)
        # stays at registry NOMINAL_PARAMS.
    },
}

DESIGNS = {'A': DESIGN_A, 'B': DESIGN_B}


# ===========================================================================
# Pure transform
# ===========================================================================
def apply_design(name, param_names, param_bounds, nominal):
    """Return (active_idx, param_bounds_new, nominal_new) for a burst design.

    Parameters
    ----------
    name : 'A' or 'B'
    param_names : list[str]            the driver's PARAM_NAMES (length D)
    param_bounds : (D,2) float ndarray the driver's PARAM_BOUNDS (natural units)
    nominal : (D,) float ndarray       the driver's NOMINAL_PARAMS (natural units)

    Returns
    -------
    active_idx : sorted list[int]      axes to SWEEP (drawn from the prior)
    param_bounds_new : (D,2) ndarray   narrowed swept bounds; point-pinned freezes
    nominal_new : (D,) ndarray         burst-favourable freeze values applied

    Raises on any swept bound or freeze value that escapes the production box,
    so a typo can never push the sweep outside the validated registry.
    """
    if name not in DESIGNS:
        raise ValueError(f"unknown burst design {name!r}; choices {sorted(DESIGNS)}")
    design = DESIGNS[name]
    idx = {n: i for i, n in enumerate(param_names)}

    pb = np.array(param_bounds, dtype=np.float64).copy()
    nom = np.array(nominal, dtype=np.float64).copy()
    prod = np.array(param_bounds, dtype=np.float64)   # immutable reference box

    def _check_inside(axis, lo, hi):
        i = idx[axis]
        plo, phi = prod[i]
        if lo < plo - 1e-9 or hi > phi + 1e-9:
            raise ValueError(
                f"design {name}: axis {axis} range [{lo},{hi}] escapes "
                f"production bounds [{plo},{phi}]")

    # 1) narrow the swept axes
    for axis, (lo, hi) in design['sweep_bounds'].items():
        if axis not in idx:
            raise KeyError(f"design {name}: unknown axis {axis!r}")
        if hi < lo:
            raise ValueError(f"design {name}: axis {axis} has hi<lo")
        _check_inside(axis, lo, hi)
        pb[idx[axis]] = (lo, hi)

    # 2) apply burst-favourable freezes (and point-pin their bounds)
    for axis, val in design['freeze_values'].items():
        if axis not in idx:
            raise KeyError(f"design {name}: unknown axis {axis!r}")
        if axis in design['sweep_bounds']:
            raise ValueError(
                f"design {name}: axis {axis} is both swept and frozen")
        _check_inside(axis, val, val)
        nom[idx[axis]] = val
        pb[idx[axis]] = (val, val)

    active_idx = sorted(idx[a] for a in design['sweep_bounds'])
    return active_idx, pb, nom


# ===========================================================================
# Driver patch (apply by hand to HPC_main_sweep.py main(), after active_idx
# is resolved -- around line 1296). Guarded so the production path is
# byte-identical when --burst_design is absent.
# ===========================================================================
PATCH_SNIPPET = r'''
# --- in the argument parser (near --sweep_group) ---------------------------
p.add_argument('--burst_design', default=None, choices=['A', 'B'],
               help="Burst-validation design from burst_validation_box.py. "
                    "A: sweep only adaptation, freeze synapses burst-favourably. "
                    "B: joint burst-permissive neuron+synapse sweep. "
                    "Overrides --sweep_group and narrows the prior box.")

# --- in main(), immediately AFTER:  active_idx = resolve_sweep_group(...) ---
if args.burst_design:
    import burst_validation_box as bvb
    active_idx, _pb, _nom = bvb.apply_design(
        args.burst_design, PARAM_NAMES, PARAM_BOUNDS, NOMINAL_PARAMS)
    PARAM_BOUNDS[:, :]      = _pb            # in place: keep module refs valid
    NOMINAL_PARAMS[:]       = _nom
    PARAM_BOUNDS_THETA[:, :] = _compute_param_bounds_theta()  # uses LOG_PARAMS
    print(f"[main] burst_design={args.burst_design}: sweeping "
          f"{[PARAM_NAMES[i] for i in active_idx]}")
# (inactive_idx / nominal_theta are computed from active_idx just below, so they
#  pick up the override automatically.)
'''


# ===========================================================================
# Smoke test -- validates BOTH designs against the REAL 36-axis registry
# (fixtures embedded so the test needs only numpy, not the heavy driver).
# ===========================================================================
def _real_registry():
    """The production registry fixtures (PARAM_NAMES / PARAM_BOUNDS /
    NOMINAL_PARAMS / LOG_PARAMS), copied verbatim from HPC_single_run.py +
    HPC_main_sweep.py so the test checks designs against the TRUE box."""
    names = [
        'Sigma', 'gbarA', 'EC50_ampa', 'EC50_nmda', 'tauA', 'U_0_ar', 'U_max',
        'U_0_sr', 'Omega_f_sr', 'Omega_f_ar', 'Omega_d', 'alpha_syn', 'DeltaT',
        'VT', 'g_ampa', 'g_nmda', 'delta_gA', 'x0', 'O_G', 'Omega_G', 'O_beta',
        'O_3K', 'Omega_5P', 'I_bias', 'F', 'I_Theta', 'omega_I', 'C_Theta',
        'U_A', 'G_T', 'gL', 'VA', 'DeltaA', 'VR', 'I_inj', 'Cm']
    bounds = np.array([
        (1.0, 15.0), (0.01, 10.0), (2.0, 150.0), (0.5, 50.0), (10.0, 1000.0),
        (1e-4, 0.05), (0.1, 1.0), (0.1, 1.0), (0.1, 4.5), (0.1, 4.5),
        (0.1, 4.5), (0.1, 1.0), (2.0, 2.0), (-51.0, -51.0), (0.05, 5.0),
        (0.01, 1.0), (0.001, 4.0), (0.05, 0.5), (0.1, 10.0), (1e-3, 1e-1),
        (0.1, 5.0), (1.0, 15.0), (0.01, 1.0), (0.3, 1.5), (0.1, 10.0),
        (0.1, 1.0), (0.01, 0.5), (0.1, 2.0), (0.1, 0.9), (50.0, 1000.0),
        (1.15, 1.15), (-73.0, -38.0), (0.1, 15.0), (-68.2, -48.2), (2.0, 7.5),
        (9.2, 34.5)], dtype=np.float64)
    nominal = np.array([
        4.0, 0.9, 8.6, 3.0, 200.0, 0.003, 0.5, 0.15, 2.0, 1.42857, 2.0, 1.0,
        2.0, -51.0, 0.3, 0.3, 0.135, 0.2, 1.5, 0.00833, 1.0, 4.5, 0.1, 0.8,
        2.0, 0.3, 0.05, 0.5, 0.6, 200.0, 1.15, -48.0, 5.0, -58.2, 4.0, 17.25])
    # LOG_PARAMS: positive-bounded axes spanning >= 1 decade
    log_idx = {k for k in range(len(names))
               if bounds[k, 0] > 0 and bounds[k, 1] > 0
               and np.log10(bounds[k, 1] / bounds[k, 0]) >= 1.0}
    return names, bounds, nominal, log_idx


def _theta_roundtrip_ok(nominal, bounds, log_idx):
    """natural -> theta -> natural must reproduce nominal exactly (ln on log axes)."""
    th = nominal.copy()
    for k in log_idx:
        th[k] = np.log(nominal[k])
    nat = th.copy()
    for k in log_idx:
        nat[k] = np.exp(th[k])
    return np.allclose(nat, nominal)


def smoke_test(verbose=True):
    """Checks for designs A and B (controlled against the real registry):
      1. apply_design runs and returns a (D,2) / (D,) shaped result
      2. active set == declared swept axes (by name)
      3. every swept axis bound lies INSIDE production bounds (narrowing only)
      4. every freeze value lies inside production bounds
      5. no axis is both swept and frozen
      6. frozen burst-favourable axes are NOT active
      7. theta round-trip holds on the modified nominal
      8. design A sweeps exactly the 3 adaptation axes
    """
    names, bounds, nominal, log_idx = _real_registry()
    D = len(names)
    idx = {n: i for i, n in enumerate(names)}
    ok = True

    def check(label, cond, detail=''):
        nonlocal ok
        ok = ok and bool(cond)
        if verbose:
            print(f"  [{'PASS' if cond else 'FAIL'}] {label}  {detail}")

    for dn in ('A', 'B'):
        design = DESIGNS[dn]
        active, pb, nom = apply_design(dn, names, bounds, nominal)

        check(f"{dn}: shapes", pb.shape == (D, 2) and nom.shape == (D,))

        want_active = sorted(idx[a] for a in design['sweep_bounds'])
        check(f"{dn}: active set == declared", active == want_active)

        inside = all(bounds[i, 0] - 1e-9 <= pb[i, 0] <= pb[i, 1] <= bounds[i, 1] + 1e-9
                     for i in active)
        check(f"{dn}: swept bounds inside production box", inside)

        frozen_ok = all(bounds[idx[a], 0] - 1e-9 <= v <= bounds[idx[a], 1] + 1e-9
                        for a, v in design['freeze_values'].items())
        check(f"{dn}: freeze values inside production box", frozen_ok)

        overlap = set(design['sweep_bounds']) & set(design['freeze_values'])
        check(f"{dn}: no swept-and-frozen overlap", not overlap, str(overlap))

        frozen_not_active = all(idx[a] not in active
                                for a in design['freeze_values'])
        check(f"{dn}: frozen axes not active", frozen_not_active)

        check(f"{dn}: theta round-trip", _theta_roundtrip_ok(nom, pb, log_idx))

    aA, _, _ = apply_design('A', names, bounds, nominal)
    check("A: sweeps exactly {delta_gA, tauA, gbarA}",
          set(names[i] for i in aA) == {'delta_gA', 'tauA', 'gbarA'},
          str([names[i] for i in aA]))

    # negative control: a deliberately out-of-box value must RAISE
    bad = {'sweep_bounds': {'tauA': (10.0, 5000.0)}, 'freeze_values': {}}
    DESIGNS['_bad'] = bad
    raised = False
    try:
        apply_design('_bad', names, bounds, nominal)
    except ValueError:
        raised = True
    del DESIGNS['_bad']
    check("escape-detection raises on out-of-box range", raised)

    if verbose:
        print(f"\nsmoke_test: {'ALL PASS' if ok else 'FAILURES PRESENT'}")
    return ok


def print_design_table(name):
    """Human-readable dump of a design (swept ranges + frozen values)."""
    d = DESIGNS[name]
    print(f"=== DESIGN {name} ===")
    print("  SWEPT axes:")
    for a, (lo, hi) in d['sweep_bounds'].items():
        print(f"    {a:12s} [{lo}, {hi}]")
    print("  FROZEN (burst-favourable) axes:")
    for a, v in d['freeze_values'].items():
        print(f"    {a:12s} = {v}")


if __name__ == '__main__':
    import sys
    if '--table' in sys.argv:
        print_design_table('A'); print(); print_design_table('B')
        sys.exit(0)
    sys.exit(0 if smoke_test() else 1)
