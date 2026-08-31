# MEAMA -- MEA Manifold Analysis

Self-contained working folder for the Full-mode tripartite campaign whose
output feeds (a) SBI posterior fitting against control and pathological MEA
recordings and (b) manifold analysis of the resulting summary-statistic space.

**Everything lives in this one folder** -- launch code, the pipeline it calls,
and the output it produces:

```
MEAMA/
  HPC_main_sweep.py, HPC_single_run.py, ASD_fun_BD_cpp.py   pipeline (patched:
                                                              37-D registry,
                                                              O_N, contact gate)
  aggregate_sweep.py, burst_metrics.py, burst_plots.py      aggregation
  seed_alloc.py                                             disjoint seed ranges
  synapse_pdist.csv                                         Sholl bouton placement
  smoke_test_param_recording.py, smoke_test_stoa_gate.py    regression checks
  launch_campaign.sh, submit_sweep_mixed.sh                 the rho=1300 campaign
                                                              (kept for reference;
                                                              not used by MEAMA)
  launch_bench.sh, bench_sizing.sh, bench_report.py         sizing benchmark
  launch_MEAMA.sh, submit_MEAMA.sh                          the MEAMA campaign
  README_MEAMA.md                                           this file
  bench_out/                                                benchmark timing runs
                                                              (never named campaign_*,
                                                              so nothing globs it in)
  campaign_meama_rho1600_full_v1/                           campaign output, once launched
  artifacts/seed_ledger.tsv                                 seed disjointness ledger
```

`launch_bench.sh` and `launch_MEAMA.sh` resolve their own directory
(`REPO_DIR`) and require the pipeline files to sit *there*, not in a parent
directory -- if `HPC_main_sweep.py` isn't in this folder, both refuse to run
with an explicit error naming what's missing. `seed_alloc.py`'s ledger and
`aggregate_sweep.py`'s glob both operate on this same folder, so nothing here
depends on where `MEAMA/` sits relative to any other checkout.


## Files

| File | Runs on | Purpose |
|---|---|---|
| `launch_bench.sh` | login node | submits the sizing benchmark (one job, six configs) |
| `bench_sizing.sh` | compute node | the benchmark itself; do not `qsub` directly |
| `bench_report.py` | login node | parses the benchmark, prints campaign sizing |
| `launch_MEAMA.sh` | login node | submits the campaign across queues |
| `submit_MEAMA.sh` | compute node | the campaign worker; do not `qsub` directly |

## Order of operations

```bash
cd /davinci-1/home/ldellamea/ANN/Phenomenological/MEAMA

# 0. prerequisites (the O_N port + contact gate must already be in place)
python -c "import HPC_main_sweep as S; assert S.N_DIMS==37; \
           assert len(S.resolve_sweep_group('tripartite'))==32; print('prereqs OK')"

# 1. size it  (~2-4 h wall; 6 configs x 2 topologies x 12 workers x 3 params)
bash launch_bench.sh
#    single-config smoke first, if you prefer:
#    QUEUE=intel NCPUS=8 C_MAX_LIST=300 SIMTIME_LIST=200 N_TOPO_BENCH=1 \
#        K_BENCH=2 N_BENCH_WORKERS=4 bash launch_bench.sh

# 2. read the sizing
python bench_report.py $(ls -td bench_out/bench_* | head -1) \
       --workers 192 --hours 20 --target 1000000

# 3. transcribe C_MAX, SIMTIME, N_TOPOLOGIES_WORKER, K_PARAMS into
#    launch_MEAMA.sh, then flip SIZING_CONFIRMED=1

# 4. dry run, read every qsub line, then launch
DRYRUN=1 bash launch_MEAMA.sh
bash launch_MEAMA.sh

# 5. monitor
python aggregate_sweep.py campaign_meama_rho1600_full_v1 --write-index
```

`launch_MEAMA.sh` refuses to submit while `SIZING_CONFIRMED=0`. That gate is
there because the inherited 80-topologies-per-task figure came from an
800 s/sim *Neuronal-mode* assumption that does not describe a tripartite
network with astrocytes, gap junctions and gliotransmission.

## Population

Fixed **total** cell density, split by astrocyte fraction; both densities are
derived from one pair of numbers so the split cannot drift:

```
rho_neuron = RHO_TOTAL * (1 - ASTRO_FRAC)      ASTRO_FRAC = 0.5  ->  Na = Nn
rho_astro  = RHO_TOTAL * ASTRO_FRAC
N          = round(rho * (C_MAX/1000)^2)
```

At `RHO_TOTAL=1600`, `ASTRO_FRAC=0.5`:

| C_MAX [um] | area [mm^2] | Nn | Na | total |
|---|---|---|---|---|
| 300 | 0.0900 | 72 | 72 | 144 |
| 350 | 0.1225 | 98 | 98 | 196 |
| 400 | 0.1600 | 128 | 128 | 256 |

If "astro:neuron = 50%" instead means `Na/Nn = 0.5` at the same total, set
`ASTRO_FRAC=0.3333333`. That is a different network: more neurons, so roughly
1.8x the synapses and a materially higher per-simulation cost.

## Single source of truth

Every quantity that appears in both the launcher and the worker
(`N_TOPOLOGIES`, `N_PARAMS_PER_WORKER`, `C_MAX`, `SIMTIME`, both densities,
mode, group, rule, gate bounds) is set **only** in `launch_MEAMA.sh` and passed
with `qsub -v`. The worker's values are inert defaults. The
"MUST match the other file" drift that the older pair of scripts was exposed to
is structurally impossible here.

`REPO_DIR` is resolved from the launcher's own path and injected. It cannot be
recovered inside a running PBS job, because PBS spools the job script to a
scratch directory -- `${BASH_SOURCE[0]}` there points at the spool copy, not at
this repository.

## Environment escape hatch

`MEAMA_NO_ENV=1` skips `module load` / `conda activate` for verifying a script
inside an already-active environment. The launchers never set it, so a
scheduled job always takes the normal path.
