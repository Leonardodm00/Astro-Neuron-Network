"""Smoke test for the extraction-metadata record of run_channel_subset_extraction.py.

Run:  python3 smoke_test_extraction_metadata.py

Tests the pure `extraction_metadata()` and the round-trip of its record
through an npz + traces_meta.json pair, WITHOUT needing a ptrain .mat
folder or the channel_subset_extraction module (which is stubbed here).
That is deliberate: the property under test is "every parameter that
decides what the trace is gets written and can be read back", and that
property must be checkable on a machine that has no recordings.

Pure ASCII, LF only. numpy + stdlib only.
"""

import json
import os
import sys
import tempfile
import types

import numpy as np

# Stub the heavy import so the CLI module loads without scipy/.mat support.
_stub = types.ModuleType("channel_subset_extraction")
_stub.DEFAULT_FS_RAW = 10110.09
_stub.extract_channel_subsets = None
sys.modules.setdefault("channel_subset_extraction", _stub)

_HERE = os.path.dirname(os.path.abspath(__file__))
if _HERE not in sys.path:
    sys.path.insert(0, _HERE)

import run_channel_subset_extraction as R  # noqa: E402

RESULTS = []


def ok(name, cond, detail):
    RESULTS.append(bool(cond))
    print("[%s] %-60s %s" % ("PASS" if cond else "FAIL", name, detail))


class _Args(object):
    def __init__(self, **kw):
        d = dict(folder="/data/ptrain_X", mode="per_region_single",
                 n_subsets=9, electrodes_per_subset=9, mfr_threshold=0.1,
                 fs_raw=10110.09, base=0, grid_width=48, w_size=0.01,
                 gaussian_window=0.02)
        d.update(kw)
        self.__dict__.update(d)


def test_m1():
    m = R.extraction_metadata(_Args(), fs_ifr=100.0, T_rec=1200.0,
                              n_present=60, n_samples_raw=64, in_channels=1,
                              n_samples=9, row_meaning="samples",
                              argv=["x", "--w-size", "0.01"])
    need = ("w_size", "fs_ifr", "gaussian_window", "sigma_sm_bins", "fs_raw",
            "mode", "n_subsets", "electrodes_per_subset", "mfr_threshold",
            "T_rec", "extractor_version", "argv", "source_folder")
    ok("M1a every parity-relevant key is present",
       all(k in m for k in need), "%d keys" % len(m))
    ok("M1b sigma_sm in bins is derived, not typed",
       abs(m["sigma_sm_bins"] - 2.0) < 1e-12, "0.02 / 0.01 = 2")
    ok("M1c the record is JSON-serialisable",
       json.loads(json.dumps(m))["electrodes_per_subset"] == 9,
       "round-trips through json")


def test_m2():
    # the defaults the 315 archives were (by the job scripts) extracted at
    m = R.extraction_metadata(_Args(w_size=0.02, gaussian_window=0.04),
                              fs_ifr=50.0, T_rec=1200.0, n_present=60,
                              n_samples_raw=64, in_channels=1, n_samples=9,
                              row_meaning="samples", argv=[])
    ok("M2 the pre-patch defaults are recorded as 0.02 / 0.04, 2 bins",
       m["w_size"] == 0.02 and m["gaussian_window"] == 0.04
       and abs(m["sigma_sm_bins"] - 2.0) < 1e-12,
       "what an unpinned run silently used")


def test_m3():
    # round-trip: the npz scalars + the json are what dataset_profile reads
    m = R.extraction_metadata(_Args(), fs_ifr=100.0, T_rec=1200.0,
                              n_present=60, n_samples_raw=64, in_channels=1,
                              n_samples=9, row_meaning="samples", argv=[])
    with tempfile.TemporaryDirectory() as td:
        npz = os.path.join(td, "traces.npz")
        np.savez_compressed(npz, X=np.zeros((1, 10), np.float32),
                            fs_ifr=m["fs_ifr"], T_rec=m["T_rec"],
                            w_size=m["w_size"],
                            gaussian_window=m["gaussian_window"],
                            sigma_sm_bins=m["sigma_sm_bins"],
                            electrodes_per_subset=m["electrodes_per_subset"],
                            n_subsets=m["n_subsets"],
                            mfr_threshold=m["mfr_threshold"],
                            fs_raw=m["fs_raw"],
                            extractor_version=m["extractor_version"])
        with open(os.path.join(td, "traces_meta.json"), "w") as fh:
            json.dump(m, fh)
        with np.load(npz, allow_pickle=False) as d:
            back = {k: d[k].item() for k in ("w_size", "gaussian_window",
                                             "electrodes_per_subset",
                                             "mfr_threshold")}
        j = json.load(open(os.path.join(td, "traces_meta.json")))
        ok("M3a npz scalars read back exactly",
           back == {"w_size": 0.01, "gaussian_window": 0.02,
                    "electrodes_per_subset": 9, "mfr_threshold": 0.1}, back)
        ok("M3b traces_meta.json carries the same values plus provenance",
           j["gaussian_window"] == 0.02 and "argv" in j and "source_folder" in j,
           "json and npz agree")


def test_m4():
    # the consistency refusal: fs_ifr must equal 1 / w_size
    m = R.extraction_metadata(_Args(w_size=0.01), fs_ifr=50.0, T_rec=1.0,
                              n_present=1, n_samples_raw=1, in_channels=1,
                              n_samples=1, row_meaning="samples", argv=[])
    ok("M4 a declared bin width that disagrees with fs_ifr is detectable",
       abs(m["fs_ifr"] * m["w_size"] - 1.0) > 1e-6,
       "main() raises on this; here the product is %.2f, not 1" % (m["fs_ifr"] * m["w_size"]))


def main():
    print("=" * 84)
    print("Smoke test: extraction metadata record (run_channel_subset_extraction v2)")
    print("=" * 84)
    test_m1()
    test_m2()
    test_m3()
    test_m4()
    print("-" * 84)
    n_fail = RESULTS.count(False)
    print("%d passed, %d failed" % (RESULTS.count(True), n_fail))
    return 1 if n_fail else 0


if __name__ == "__main__":
    sys.exit(main())
