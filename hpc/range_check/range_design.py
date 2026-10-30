#!/usr/bin/env python3
"""
range_design.py -- the designs of the standard prior range calibration
(specs/SPEC.md Block 6, decided by the author on 2026-10-30). Pure functions of
their inputs: numpy only, no file I/O, no pipeline import.

Design dimensions: the free run-level axes of a setup, in the sweep's theta
coordinate (ln on log-uniform axes), each with its box [lo_j, hi_j], plus
conn_prob in natural units, as the last dimension, when the setup draws it per
topology (flat rule, lo < hi).

corners   two-level design X in {-1, +1}^(N x d): -1 stands for the box's lower
          bound, +1 for the upper one. The full factorial (N = 2^d) when
          2^d <= budget; otherwise a regular 2^(d-p) fraction of N = 2^b runs,
          2^b the largest power of two <= budget, whose p = d - b generators
          are chosen for the highest resolution R found, R >= 5: no main
          effect or two-axis interaction is aliased with another main effect
          or two-axis interaction.
interior  cells of the B-bin grid. Each dimension's box is cut into B bins of
          equal width in its own coordinate, so each bin holds the same prior
          mass under the setup's uniform or log-uniform prior; a cell takes
          the bin midpoints lo + (i + 1/2)(hi - lo)/B, with the bin index i
          drawn uniformly and independently in every dimension.

Custom code: the fraction's generator search. pyDOE3 1.6.2 was checked on
2026-10-30: fracfact_by_res(14, 5) returned 64 runs with 72 pairs of two-axis
interactions fully aliased, and fracfact_by_res(23, 5) 64 runs with main effects
aliased with two-axis interactions, so neither is resolution V; fracfact_opt
offers no run budget. The search below is the textbook construction (base
factors from the bits of the run index, each further factor the product of a
subset of them) with a greedy choice of subsets; min_word_length verifies the
result independently.
"""
import itertools
import math

import numpy as np

CORNER_BUDGET = 16384
MIN_RESOLUTION = 5


# ---------------------------------------------------------------------------
# two-level designs
# ---------------------------------------------------------------------------
def popcount_table(b):
    """Number of set bits of every b-bit mask, as an int array of length 2^b."""
    t = np.zeros(1 << b, dtype=np.int64)
    for k in range(b):
        t += (np.arange(1 << b) >> k) & 1
    return t


def full_factorial(d):
    """All 2^d sign vectors: row r, column k is +1 if bit k of r is 0, else -1."""
    r = np.arange(1 << d)[:, None]
    return (1 - 2 * ((r >> np.arange(d)[None, :]) & 1)).astype(np.int8)


def regular_fraction(b, generators):
    """2^b runs: b base columns (full_factorial(b)) followed by one column per
    generator mask, the product of the base columns whose bits the mask sets."""
    base = full_factorial(b)
    r = np.arange(1 << b)
    pop = popcount_table(b)
    extra = [(1 - 2 * (pop[r & g] & 1)).astype(np.int8) for g in generators]
    return np.column_stack([base] + extra) if extra else base


def min_word_length(generators, up_to):
    """Shortest word of the defining relation among the products of at most
    `up_to` generators: |G| + popcount(xor of their masks). Words of more than
    `up_to` generators have length > up_to, so a result >= up_to + 1 means
    every word is at least that long. Returns up_to + 1 if no shorter word."""
    best = up_to + 1
    for s in range(1, min(up_to, len(generators)) + 1):
        for G in itertools.combinations(generators, s):
            x = 0
            for g in G:
                x ^= g
            best = min(best, s + bin(x).count('1'))
    return best


def fraction_generators(b, p, resolution, rng, attempts=8):
    """p generator masks over b base factors such that every word of the
    defining relation has length >= resolution, or None if the greedy search
    finds none in `attempts` random orders.

    A word made of the generators G and the base factors in xor(G) has length
    |G| + popcount(xor(G)). Words of >= resolution generators are long enough
    by their generators alone, so a candidate m is allowed iff for every subset
    G of the accepted generators with |G| = s <= resolution - 2,
    popcount(m ^ xor(G)) >= resolution - 1 - s; each accepted generator
    therefore removes the Hamming balls of radius resolution - 2 - s around the
    new values xor(G + {g})."""
    R = int(resolution)
    pop = popcount_table(b)
    balls = {rad: np.flatnonzero(pop <= rad) for rad in range(0, R)}
    for _ in range(attempts):
        allowed = pop >= R - 1                       # the word {g} alone
        xors = {0: np.zeros(1, dtype=np.int64)}
        for s in range(1, R - 1):
            xors[s] = np.zeros(0, dtype=np.int64)
        accepted = []
        for m in rng.permutation(1 << b):
            if len(accepted) == p:
                break
            if not allowed[m]:
                continue
            accepted.append(int(m))
            for s in range(R - 2, 0, -1):            # largest first: uses the old xors[s - 1]
                new = xors[s - 1] ^ int(m)
                xors[s] = np.concatenate([xors[s], new])
                rad = R - 2 - s
                allowed[(new[:, None] ^ balls[rad][None, :]).ravel()] = False
        if len(accepted) == p and min_word_length(accepted, R - 1) >= R:
            return accepted
    return None


def two_level_design(d, budget=CORNER_BUDGET, rng=None, max_resolution=8):
    """(X, info): the corner design for d dimensions within `budget` runs.

    info: kind ('full' or 'fraction'), n_runs, resolution (None for the full
    factorial), base (b), generators (each as the list of base-factor indices
    whose product it is). Raises RuntimeError if no fraction of resolution
    >= MIN_RESOLUTION is found."""
    if d < 1:
        raise ValueError('need at least one dimension')
    if (1 << d) <= budget:
        return full_factorial(d), dict(kind='full', n_runs=1 << d, resolution=None, base=d,
                                       generators=[])
    rng = rng if rng is not None else np.random.default_rng(0)
    b = int(math.floor(math.log2(budget)))
    p = d - b
    for R in range(max_resolution, MIN_RESOLUTION - 1, -1):
        gens = fraction_generators(b, p, R, rng)
        if gens is not None:
            X = regular_fraction(b, gens)
            return X, dict(kind='fraction', n_runs=1 << b, resolution=R, base=b,
                           generators=[[k for k in range(b) if (g >> k) & 1] for g in gens])
    raise RuntimeError('no two-level fraction of %d factors in %d runs with resolution >= %d found'
                       % (d, 1 << b, MIN_RESOLUTION))


def aliased_low_order_pairs(X):
    """Pairs among the main-effect and two-axis-interaction columns of X that
    are fully aliased (|correlation| = 1). Empty for resolution >= 5. For
    checks; it builds a (d + d(d-1)/2)-square Gram matrix."""
    n, d = X.shape
    cols, names = [], []
    for i in range(d):
        cols.append(X[:, i].astype(np.float64))
        names.append((i,))
    for i, j in itertools.combinations(range(d), 2):
        cols.append(X[:, i].astype(np.float64) * X[:, j])
        names.append((i, j))
    M = np.array(cols)
    G = np.abs(M @ M.T) / n
    iu = np.triu_indices(len(names), 1)
    hit = G[iu] > 1 - 1e-9
    return [(names[a], names[c]) for a, c in zip(iu[0][hit], iu[1][hit])]


# ---------------------------------------------------------------------------
# interior grid
# ---------------------------------------------------------------------------
def bin_midpoints(lo, hi, n_bins):
    """Midpoints of n_bins equal-width bins of [lo, hi]: lo + (i + 1/2)(hi - lo)/n_bins."""
    return lo + (np.arange(n_bins) + 0.5) * (hi - lo) / n_bins


def interior_cells(n, d, n_bins, rng):
    """n x d bin indices, each uniform on {0, ..., n_bins - 1}, independent."""
    return rng.integers(0, n_bins, size=(n, d))


def bin_of(value, lo, hi, n_bins, tol=1e-9):
    """The bin whose midpoint equals `value` (within tol of the box width), or -1."""
    if not hi > lo:
        return -1
    i = int(round((value - lo) / (hi - lo) * n_bins - 0.5))
    if 0 <= i < n_bins and abs(value - (lo + (i + 0.5) * (hi - lo) / n_bins)) <= tol * (hi - lo):
        return i
    return -1


def corner_code(value, lo, hi, tol=1e-9):
    """-1 at the lower bound, +1 at the upper, 0 at the centre, None otherwise."""
    w = tol * max(hi - lo, 1e-300)
    if abs(value - lo) <= w:
        return -1
    if abs(value - hi) <= w:
        return 1
    if abs(value - 0.5 * (lo + hi)) <= w:
        return 0
    return None


# ---------------------------------------------------------------------------
# splitting a design over topologies and workers
# ---------------------------------------------------------------------------
def chunk_plan(n_points, n_topologies, n_workers):
    """(k, n_pad): runs per worker per topology so that n_topologies x
    n_workers x k >= n_points, and the padding that fills the last slots."""
    k = max(1, int(math.ceil(n_points / float(n_topologies * n_workers))))
    return k, n_topologies * n_workers * k - n_points
