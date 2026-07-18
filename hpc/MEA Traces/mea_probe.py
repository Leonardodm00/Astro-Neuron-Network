#!/usr/bin/env python3
# mea_probe.py
# =============================================================================
# Virtual MEA probe geometry and the distance-scaling weight matrix.
#
# SEPARATION OF CONCERNS
#   This module knows ONLY about geometry and the phenomenological amplitude
#   scaling law. It has no notion of time, spikes, noise, or detection.
#
# PROBE
#   n_side x n_side square grid of square electrodes, centred on the culture.
#     - edge      : electrode side length [um]           (default 25)
#     - pitch     : centre-to-centre spacing [um]        (default 60)
#     - n_side    : electrodes per row/col               (default 3 -> 9 total)
#   Each electrode face is tiled by n_sub x n_sub recording SUB-SITES; the
#   electrode signal is the mean over its sub-sites (finite-area averaging).
#
# SCALING LAW (phenomenological; NOT a physical dipole)
#   For neuron n and sub-site s at in-plane distance d(n,s):
#       r_tilde(n,s) = max(d(n,s), r_min)
#       A(n,s)       = A_ref * (r_ref / r_tilde(n,s)) ** n_dec
#   The per-(neuron, electrode) weight is the sub-site mean:
#       W(n,e) = mean_{s in S_e} A(n,s)
#   Sub-site averaging is linear, so it is folded into W here, once per
#   topology. n_dec = 2 (DEFAULT) gives a dipole-like 1/r^2 intensity decay;
#   n_dec = 1 reproduces the inverse-distance (1/r) law of the original
#   Virtual_electrodes.py. This is a phenomenological amplitude scaling law,
#   NOT a physical current-dipole field (no conductivity / dipole length).
#
# NOTE ON KD-TREES
#   For only n_side^2 electrodes (9) x n_sub^2 sub-sites (16) = 144 points, a
#   dense vectorised distance to all Nn neurons is O(144 * Nn) and is both
#   simpler and faster than a KDTree. The KDTree in the old code is therefore
#   dropped. r_max is used purely as a per-neuron REACH cutoff to skip
#   neurons whose contribution is below the noise floor at synthesis time.
#
# UNITS: distances in um, amplitudes in uV (or in sigma-units if A_ref is
#        given in sigma-units; the module is unit-agnostic in A_ref).
# =============================================================================

from dataclasses import dataclass, field

import numpy as np


@dataclass
class ProbeConfig:
    n_side: int = 3          # electrodes per side (3 -> 9 electrodes)
    pitch: float = 60.0      # electrode centre-to-centre [um]
    edge: float = 25.0       # electrode side length [um]
    n_sub: int = 4           # sub-sites per side (n_sub^2 per electrode)


@dataclass
class ScalingConfig:
    A_ref: float = 75.0      # reference amplitude at r_ref [uV] (= SNR_ref * sigma)
    r_ref: float = 30.0      # reference distance [um]
    r_min: float = 10.0      # near-field floor [um] (soma scale; kills blow-up)
    n_dec: float = 2.0       # decay exponent (2 = dipole-like 1/r^2; 1 = 1/r)


def make_probe(c_max, cfg: ProbeConfig = None):
    """
    Build a centred n_side x n_side probe for a square culture [0, c_max]^2.

    Returns dict:
      centers   : (E, 2) electrode centre coordinates [um], E = n_side^2
      sub_sites : (E, n_sub^2, 2) sub-site coordinates [um]
      cfg       : the ProbeConfig used
      center_xy : (2,) culture centre used = (c_max/2, c_max/2)
    """
    if cfg is None:
        cfg = ProbeConfig()
    cx = cy = 0.5 * c_max

    # Electrode centre offsets, symmetric about the culture centre.
    k = cfg.n_side
    offs = (np.arange(k) - (k - 1) / 2.0) * cfg.pitch      # e.g. [-60, 0, 60]
    cc = []
    for iy in range(k):
        for ix in range(k):
            cc.append((cx + offs[ix], cy + offs[iy]))
    centers = np.asarray(cc, dtype=float)                  # (E, 2)

    # Sub-site offsets within one electrode face (n_sub x n_sub grid of
    # sub-site centres spanning the edge, symmetric about the electrode centre).
    ns = cfg.n_sub
    sub_step = cfg.edge / ns
    sub_off = (np.arange(ns) - (ns - 1) / 2.0) * sub_step  # centred, within +/- edge/2
    sub_grid = []
    for dy in sub_off:
        for dx in sub_off:
            sub_grid.append((dx, dy))
    sub_grid = np.asarray(sub_grid, dtype=float)           # (n_sub^2, 2)

    sub_sites = centers[:, None, :] + sub_grid[None, :, :]  # (E, n_sub^2, 2)
    return dict(centers=centers, sub_sites=sub_sites, cfg=cfg,
                center_xy=np.array([cx, cy]))


def compute_weights(neuron_xy, probe, scfg: ScalingConfig = None):
    """
    Per-(neuron, electrode) amplitude weight W(n,e) from the scaling law,
    with sub-site averaging folded in.

    neuron_xy : (Nn, 2) neuron positions [um]
    Returns   : W (Nn, E) [same units as A_ref]
    """
    if scfg is None:
        scfg = ScalingConfig()
    xy = np.asarray(neuron_xy, dtype=float)
    sub = probe['sub_sites']                                # (E, S, 2)
    E = sub.shape[0]
    Nn = xy.shape[0]
    W = np.empty((Nn, E), dtype=float)
    for e in range(E):
        # (Nn, S, 2) -> (Nn, S) distances
        diff = xy[:, None, :] - sub[e][None, :, :]
        d = np.sqrt(np.einsum('nsc,nsc->ns', diff, diff))
        r_tilde = np.maximum(d, scfg.r_min)
        A = scfg.A_ref * (scfg.r_ref / r_tilde) ** scfg.n_dec
        W[:, e] = A.mean(axis=1)                            # sub-site mean
    return W


def r_max_reach(scfg: ScalingConfig, gamma, sigma_noise):
    """
    Distance at which a unit's amplitude falls to gamma * sigma_noise:
        r_max = r_ref * (A_ref / (gamma * sigma_noise)) ** (1 / n_dec)
    Used as a per-neuron reach cutoff. Returns a scalar [um].
    """
    ratio = scfg.A_ref / (gamma * sigma_noise)
    return scfg.r_ref * ratio ** (1.0 / scfg.n_dec)


def reach_mask(W, floor):
    """
    Per-neuron boolean: True if the neuron's weight to ANY electrode exceeds
    `floor` (else it contributes below the noise floor everywhere and can be
    skipped during synthesis). Returns (Nn,) bool.
    """
    return W.max(axis=1) >= floor


if __name__ == '__main__':
    # Minimal self-print for a 1100 um culture.
    pr = make_probe(1100.0)
    print('electrode centres [um]:')
    print(pr['centers'])
    print('sub-sites per electrode:', pr['sub_sites'].shape[1])
    print('centre:', pr['center_xy'])
