"""CLI driver -- extract channel-subset IFR traces from a ptrain folder and save.

Runs the full extractor (load -> geometry/MFR -> partition -> IFR) on ONE folder
of ptrain_<idx>.mat files and writes a self-describing .npz plus (for the
partition modes) the electrode-map and per-subregion IFR PNGs.

Usage
-----
    python3 run_channel_subset_extraction.py FOLDER --out-dir OUT \
        [--mode multichannel|per_region_single|whole_culture] \
        [--n-subsets 9] [--electrodes-per-subset 9] [--mfr-threshold 0.1] \
        [--fs-raw 10110.09] [--base 0] [--no-plots]

Output (in OUT)
---------------
    traces.npz : arrays
        X            : (rows, K) float32 IFR traces
        row_meaning  : "channels" (multichannel: rows are channels of ONE sample)
                       or "samples" (per_region_single / whole_culture: rows are
                       independent single-channel samples)
        in_channels  : C for multichannel, else 1
        n_samples    : number of training samples this recording yields
        fs_ifr, mode, index_base, grid_width, T_rec, n_samples_raw
        centers, center_mfr, discarded  (empty for whole_culture)
    subregion_map.png, subregion_ifrs.png  (unless --no-plots / whole_culture)

Only numpy / scipy / matplotlib are required (no torch): this is data extraction,
not training.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from typing import List, Optional

import numpy as np

from channel_subset_extraction import DEFAULT_FS_RAW, extract_channel_subsets

EXTRACTOR_VERSION = "run_channel_subset_extraction/2"   # 1 = pre-2026-09-11, no metadata


def extraction_metadata(args, fs_ifr, T_rec, n_present, n_samples_raw,
                        in_channels, n_samples, row_meaning, argv=None):
    """Every parameter that decides what the trace IS, in one dict.

    [CORRECTION 2026-09-11] Until this version traces.npz recorded fs_ifr
    and T_rec but NOT gaussian_window (sigma_sm), electrodes_per_subset
    (the n_e each pooled trace is a mean over), n_subsets, mfr_threshold or
    fs_raw. The consumer (Sbi-extractor) stamps its sidecars with the
    encoder checkpoint's sigma_sm instead, so an archive smoothed at the
    0.04 s default and an export declaring 0.02 s could never be told
    apart from the files. This dict is written BOTH into the npz (so the
    archive stays self-describing) and to traces_meta.json beside it (so a
    reader that only opens JSON, and a human, can see it).

    Pure: no I/O, so the smoke test can check it without a .mat folder.
    """
    w = float(args.w_size)
    g = float(args.gaussian_window)
    return {
        "extractor_version": EXTRACTOR_VERSION,
        # preprocessing -- the parity-critical block
        "w_size": w,                       # Delta_t [s], IFR bin width
        "fs_ifr": float(fs_ifr),           # = 1 / w_size, as returned
        "gaussian_window": g,              # sigma_sm [s]
        "sigma_sm_bins": g / w,            # sigma_sm / Delta_t
        "fs_raw": float(args.fs_raw),
        # pooling geometry
        "mode": str(args.mode),
        "n_subsets": int(args.n_subsets),
        "electrodes_per_subset": int(args.electrodes_per_subset),
        "grid_width": int(args.grid_width),
        "index_base": int(args.base),
        "mfr_threshold": float(args.mfr_threshold),
        # what came out
        "T_rec": float(T_rec),
        "n_present": int(n_present),
        "n_samples_raw": int(n_samples_raw),
        "in_channels": int(in_channels),
        "n_samples": int(n_samples),
        "row_meaning": str(row_meaning),
        # provenance
        "source_folder": os.path.abspath(str(args.folder)),
        "argv": list(argv if argv is not None else sys.argv),
    }


def main(argv: Optional[List[str]] = None) -> int:
    p = argparse.ArgumentParser(
        description="Extract channel-subset IFR traces from a ptrain folder.")
    p.add_argument("folder", help="directory of ptrain_<idx>.mat files (one recording)")
    p.add_argument("--out-dir", required=True, help="output directory")
    p.add_argument("--mode", default="multichannel",
                   choices=["multichannel", "per_region_single", "whole_culture"])
    p.add_argument("--n-subsets", type=int, default=9)
    p.add_argument("--electrodes-per-subset", type=int, default=9)
    p.add_argument("--mfr-threshold", type=float, default=0.1)
    p.add_argument("--fs-raw", type=float, default=DEFAULT_FS_RAW)
    p.add_argument("--base", type=int, default=0, choices=[0, 1])
    p.add_argument("--grid-width", type=int, default=48)
    p.add_argument("--w-size", type=float, default=0.02)
    p.add_argument("--gaussian-window", type=float, default=0.04)
    p.add_argument("--no-plots", action="store_true", help="skip PNG rendering")
    args = p.parse_args(argv)

    traces, fs_ifr, diag = extract_channel_subsets(
        args.folder, mode=args.mode, n_subsets=args.n_subsets,
        electrodes_per_subset=args.electrodes_per_subset,
        mfr_threshold=args.mfr_threshold, fs_raw=args.fs_raw, index_base=args.base,
        grid_width=args.grid_width, w_size=args.w_size,
        gaussian_window=args.gaussian_window, return_diagnostics=True)

    os.makedirs(args.out_dir, exist_ok=True)

    if args.mode == "multichannel":
        X = np.asarray(traces[0], dtype=np.float32)          # (C, K), rows = channels
        row_meaning = "channels"
        in_channels = int(X.shape[0])
        n_samples = 1
    else:
        X = np.stack([np.asarray(t, dtype=np.float32).reshape(-1) for t in traces],
                     axis=0)                                  # (n_samples, K), rows = samples
        row_meaning = "samples"
        in_channels = 1
        n_samples = int(X.shape[0])

    centers = np.array([s.center for s in diag.subregions], dtype=np.int64)
    center_mfr = np.array([s.center_mfr for s in diag.subregions], dtype=np.float64)
    discarded = np.array(diag.discarded, dtype=np.int64)

    meta = extraction_metadata(args, fs_ifr, diag.T_rec, diag.n_present,
                               diag.n_samples, in_channels, n_samples,
                               row_meaning)
    if abs(meta["fs_ifr"] * meta["w_size"] - 1.0) > 1e-6:
        raise RuntimeError(
            "fs_ifr = %r from the extractor does not equal 1 / w_size = 1 / %r; "
            "refusing to write an archive whose declared bin width disagrees "
            "with its sampling rate" % (meta["fs_ifr"], meta["w_size"]))

    npz_path = os.path.join(args.out_dir, "traces.npz")
    np.savez_compressed(
        npz_path,
        X=X, row_meaning=row_meaning, in_channels=in_channels, n_samples=n_samples,
        fs_ifr=float(fs_ifr), mode=args.mode, index_base=int(diag.index_base),
        grid_width=int(diag.grid_width), T_rec=float(diag.T_rec),
        n_samples_raw=int(diag.n_samples), n_present=int(diag.n_present),
        centers=centers, center_mfr=center_mfr, discarded=discarded,
        # [2026-09-11] the preprocessing parameters, embedded so the archive
        # is self-describing (scalars; the full record is traces_meta.json)
        w_size=meta["w_size"], gaussian_window=meta["gaussian_window"],
        sigma_sm_bins=meta["sigma_sm_bins"], fs_raw=meta["fs_raw"],
        n_subsets=meta["n_subsets"],
        electrodes_per_subset=meta["electrodes_per_subset"],
        mfr_threshold=meta["mfr_threshold"],
        extractor_version=meta["extractor_version"])
    meta_path = os.path.join(args.out_dir, "traces_meta.json")
    with open(meta_path, "w") as fh:
        json.dump(meta, fh, indent=2, sort_keys=True)

    if (not args.no_plots) and diag.subregions:
        from channel_subset_viz import plot_subregion_ifrs, plot_subregion_map
        plot_subregion_map(diag, os.path.join(args.out_dir, "subregion_map.png"))
        arr = traces[0] if args.mode == "multichannel" else X
        plot_subregion_ifrs(arr, fs_ifr,
                            os.path.join(args.out_dir, "subregion_ifrs.png"),
                            centers=[s.center for s in diag.subregions])

    print("mode=%s  X.shape=%s  row_meaning=%s  in_channels=%d  n_samples=%d  fs_ifr=%.3f"
          % (args.mode, X.shape, row_meaning, in_channels, n_samples, fs_ifr))
    print("present=%d  discarded(<theta)=%d  centres=%s"
          % (diag.n_present, discarded.size, centers.tolist()))
    print("sigma_sm=%.4f s (%.2f bins)  w_size=%.4f s  n_e=%d  mfr_threshold=%.3f"
          % (meta["gaussian_window"], meta["sigma_sm_bins"], meta["w_size"],
             meta["electrodes_per_subset"], meta["mfr_threshold"]))
    print("wrote", npz_path)
    print("wrote", meta_path)
    return 0


if __name__ == "__main__":
    sys.exit(main())
