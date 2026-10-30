# SPEC -- Astro-Neuron-Network (hpc/): refractory hold, refractory audit, LindaColloDUP launch package

Single source of truth for what the covered code must do. The code follows
this file; when the two disagree, fix whichever is wrong and record why in the
decision log. The integration tester reads only the repository, so every
constraint the code must satisfy is stated here, including those that come
from the author's decisions.

Notation rules: every symbol is defined at first use; a conditional quantity
keeps its conditioning every time it appears; quantifiers ("for each neuron
i", "for all t in ...") are stated explicitly.

Drafted from the code on 2026-10-30. An entry says `Confirmed: inferred`
until the author has checked it against what was intended; a constraint that
comes from an explicit decision of the author says `Confirmed: yes` with the
date of the decision.

Last reviewed at commit: 562a9fd (the base of the commits that add this file)

## 0. Coverage

- `hpc/refractory_audit/` (the audit and its `checks/`)
- `hpc/Phenomenological_finalv1/LindaColloDUP/` (launch package and its `checks/`)
- the refractory handling of the neuron model in
  `hpc/Phenomenological_finalv1/ASD_fun_BD_cpp.py` and
  `hpc/Phenomenological_finalv1/Giulia/ASD_fun_BD_cpp.py`: in
  `Neuronal_Network`, the dV/dt line and the threshold, reset and refractory
  arguments of the neuron `NeuronGroup` -- not the rest of those files

Everything else in the repository is not specified yet and is out of scope
for the tester.

## 1. Scientific goal

The sweep pipeline simulates cultured neuron-astrocyte networks (Brian2,
cpp_standalone; CAdEx neurons of Gorski et al. 2021, Neural Comput 33:41) to
build training sets for simulation-based inference. The covered code
(a) holds each neuron's membrane potential at the reset value during its
refractory period, as the cited model defines it; (b) measures, for campaigns
already run without that hold, how many runs lost neurons to the refractory
runaway; (c) launches the neurons-only LindaColloDUP campaign on davinci.
Consumers: the author (audit verdicts, campaign outputs) and the SBI training
(campaign outputs).

## 2. Mathematical formulation

For each neuron i = 0, ..., Nn - 1 (`Neuronal_Network`):

    dV_i/dt = noise_i + [ gL (EL - V_i) + gL DeltaT exp((V_i - VT)/DeltaT)
                          + g_A,i (EA - V_i) + I_inj b_i - I_syn,i ] / Cm
    dg_A,i/dt = ( gbar_A / (1 + exp((V_A - V_i)/Delta_A)) - g_A,i ) / tau_A
    noise_i = sigma sqrt(2 gL_ref / Cm) xi_i / sqrt(dt), xi_i ~ N(0, 1) drawn
              each step and held constant over the step

Spike of neuron i at time t_s,i: V_i(t_s,i) > V_D while neuron i is not
refractory. Reset at t_s,i: V_i <- V_R and g_A,i <- g_A,i + delta_gA.
Refractory hold, for each neuron i and each of its spike times t_s,i:

    V_i(t) = V_R   for all t in (t_s,i, t_s,i + t_ref],   t_ref = 3 ms

implemented as the Brian2 flag `(unless refractory)` on the dV/dt line;
g_A,i, the synaptic variables and the noise generator keep evolving
(Gorski et al. 2021, Sec. 2: V "remains at this value during a refractory
period"). Brian2 evaluates the threshold as `(V > VD) and not_refractory`;
without the hold V_i integrates during t_ref with no threshold able to stop
it, and a neuron pushed past the unstable fixed point of the current balance
diverges to inf/NaN and never spikes again.

Refractory-runaway screen (`refractory_scan.classify_run`), for one run of
simulated duration T [s] with spike times t_k [s] and neuron indices i_k:

    count_i        = number of finite spikes k with i_k = i (0 <= i_k < Nn)
    last_i         = max over those k of t_k (-inf if count_i = 0)
    spiked_i       = (count_i > 0)
    early_i        = spiked_i and last_i <= t_early        (evaluated only if T >= 5 t_early)
    early_strict_i = spiked_i and last_i <= t_early_strict (same condition)
    tau_late       = max(late_min_s, late_frac T)
    late_i         = spiked_i and t_early < last_i <= T - tau_late and
                     #{k spiking : last_k > last_i + late_gap_s}
                         >= max(1, ceil(late_active_frac (Nn - 1)))

A run is "affected" when it has at least one early neuron (the headline,
calibrated rule) and "fully silenced early" when every spiking neuron is early.

### Notation

| Symbol | Meaning | Type / shape | Units |
|---|---|---|---|
| V_i | membrane potential of neuron i | Brian2 state, float64 | mV |
| g_A,i | adaptation conductance of neuron i | state, float64 | nS |
| V_R, V_D, VT, EL, EA, V_A | reset, detection limit (-40 mV), soft threshold, leak and adaptation reversal, adaptation half-activation | scalars per neuron | mV |
| DeltaT, Delta_A | spike-initiation and adaptation slopes | scalars | mV |
| gL, gL_ref, gbar_A, delta_gA | leak, fixed noise-reference leak, maximal subthreshold adaptation, spike-triggered increment | scalars | nS |
| Cm | membrane capacitance | scalar | pF |
| tau_A | adaptation time constant | scalar | ms |
| I_inj b_i | frozen per-neuron bias, b_i ~ U(-0.5, 0.5) drawn once at build | scalar per neuron | pA |
| I_syn,i | synaptic current (inward negative) | state | pA |
| sigma | noise amplitude (stationary SD of the passive membrane) | scalar | mV |
| t_ref | refractory period | scalar, 3 ms | ms |
| t_s,i | a spike time of neuron i | float | s |
| T | simulated time of one run | float | s |
| Nn | number of neurons | int | -- |

## 3. Data contracts

| Item | Source / format | Shape and axis order | dtype | Units | Conventions |
|---|---|---|---|---|---|
| `spk_N_t` | `topo_*/iter_*.npz` of a sweep task | (K,) | float32 | s | from t = 0 of the run; non-finite times dropped by the screen |
| `spk_N_i` | same file | (K,) | int32 | -- | 0-based neuron index, aligned with `spk_N_t` |
| `job_args.json` | task folder `sweep_<queue>_task<NNNN>/` | dict | JSON | -- | `simtime` [s], `Nn`, `lib_dir` (folder of the `ASD_fun_BD_cpp.py` the task ran) |
| `iter_*.json` | next to each npz | dict | JSON | -- | `param_names` (column names of the parameter vector) |
| `<campaign>_runs.csv` | refractory_scan output | one row per run | CSV | s for times | columns: task, topo, iter, status, T, Nn, nn_source, n_spikes, n_spiking, n_early, n_early_strict, n_late, n_suspect, frac_suspect, net_silent_early, t_last_any, conn_prob, then par_<name> |
| `<campaign>_summary.json` | refractory_scan output | dict | JSON | percent | includes runs_early, pct_runs_early, ci95_early (Wilson 95 %), runs_any_suspect, runs_late, pct_runs_net_silent_early, pct_neurons_early |

## 4. Global conventions

- Units: Brian2 units in the model; seconds in the saved spike files; mV, nS,
  pF, pA, ms in documentation and checks.
- Randomness: the pipeline's seeds are unchanged by the covered code; the
  screen's sampling uses `numpy.random.default_rng(--seed)`, default seed 1.
- Binding decisions. The project's decision log is not available where this
  file was written, so the decisions are recorded in the project doc
  `claude/LindaColloDUP_Campaign_Operations.md`:
  - 2026-10-30 [author]: fix the refractory runaway: the refractory hold of
    section 2 in every repository copy of the CAdEx network and in
    LindaColloDUP (`REFRACTORY_CLAMP=yes`). Not applied in place to the
    cluster's `MEAMA/` (its campaigns run there).
  - 2026-10-29 [author]: LindaColloDUP runs MODE Neuronal; freezes Cm,
    I_inj, VR, VA, DeltaA, EC50_ampa, EC50_nmda (and the inert alpha_syn,
    O_G, Omega_G); SIMTIME 180 s; RHO_TOTAL 1300 cells/mm^2; everything else
    as MEAMA.

## 5. Blocks

### Block 1: refractory hold in the CAdEx neuron

- Module: `hpc/Phenomenological_finalv1/ASD_fun_BD_cpp.py` and
  `hpc/Phenomenological_finalv1/Giulia/ASD_fun_BD_cpp.py`, `Neuronal_Network`.
- Change: the dV/dt line ends with ` (unless refractory)`; four comment lines
  above it. Nothing else.
- Inputs / outputs: unchanged (same parameters in; spike trains out).
- Library calls relied on: Brian2 equation flag `unless refractory`;
  `NeuronGroup(threshold='V > VD', reset='V = VR; gA += delta_gA', refractory=t_ref)`.
- Custom code: none.
- Test oracles: (1) paired runs (`hpc/refractory_audit/checks/check_refractory_blowup.py
  --allow_clamped_source`, LindaColloDUP prior, seed 2026, 4 runs x 5 s):
  0 NaN neurons in every run (the same runs as shipped: 141, 159, 159, 116 of
  159); (2) spike counts equal to those of the same runs with the flag added
  by `check_refractory_blowup.py --clamp` (exact equality: same equations,
  same seeds); (3) `apply_refractory_clamp.py --check` reports the file as
  already clamped and the audit's CODE section reports it CLAMPED.
- Data flow: every campaign launched from a copy of these files.
- Confirmed: yes 2026-10-30 (decision in section 4).
- Status: smoke-tested 2026-10-30 (skipped: the cluster's 37-D copies, which
  are not in the repository; runs longer than 5 s).

### Block 2: refractory_scan, the spike-train screen

- Module: `hpc/refractory_audit/refractory_scan.py` (numpy only).
- Public API: `classify_run(spk_t, spk_i, Nn: int, T: float, cfg: dict | None = None) -> dict`;
  `scan_campaign(campaign_dir, out_dir, sample, seed, workers, cfg)`;
  CLI `python refractory_scan.py CAMPAIGN_DIR [...] --out DIR [--sample N] [--seed S] [--workers W]`.
- Inputs: spk_t (K,) float [s]; spk_i (K,) int, 0-based; Nn; T [s];
  cfg overrides of DEFAULT_CFG.
- Outputs: classify_run returns length-Nn arrays count (int), last (float,
  s, -inf if no spike), spiked, early, early_strict, late (bool), and the
  scalars tau_late (s) and early_evaluated (bool). scan_campaign writes the
  CSV and summary files of section 3 and a text summary.
- Parameters: DEFAULT_CFG t_early = 1.0 s, t_early_strict = 0.1 s,
  late_frac = 0.5, late_min_s = 20 s, late_active_frac = 0.10,
  late_gap_s = 1.0 s; `--sample` 1000 (0 = all runs), `--seed` 1, `--workers` 4.
- Custom code: the rules themselves; no library implements this screen.
- Test oracles: `smoke_test_refractory_scan.py` (41 checks: classify_run unit
  cases, a synthetic campaign, the driver end to end with a fake HOME, the
  bytes of the six package files); calibration against Brian2's NaN record
  (`checks/calibrate_refractory_scan.py`, 5 s runs): early-rule sensitivity
  0.964, 0.964 and 0.910 in three run sets, 0 false positives among 6,531
  healthy neurons. The late rule is logic-tested only.
- Data flow: reads sweep outputs (section 3); feeds Block 3.
- Confirmed: inferred (rules and thresholds proposed in the conversation of
  2026-10-30 and calibrated; the author has not confirmed the thresholds).
- Status: smoke-tested 2026-10-30; on davinci only the smoke test has run.

### Block 3: the audit driver and its PBS job

- Module: `hpc/refractory_audit/check_refractory_nan.sh` (login node) and
  `hpc/refractory_audit/submit_refractory_scan.sh` (PBS).
- CLI: `bash check_refractory_nan.sh [--sample N] [--workers W] [--out DIR] [CAMPAIGN_DIR ...]`;
  default campaigns, under `/davinci-1/home/ldellamea/ANN/Phenomenological/`:
  `MEAMA/campaign_meama_rho1600_n01` and
  `Main/Giulia_Astro/campaign_cadex_hhgap_v1`, `_v2`, `_v4`, `_v5`.
- Outputs: `refractory_audit_<UTC>/` next to the script, holding `audit.log`
  (sections 1 CODE, 2 LOGS, 3 DATA, VERDICT) and Block 2's files per
  campaign. Reads campaigns and job logs only; writes nothing else.
- `submit_refractory_scan.sh`: `qsub -q cpu`, 16 cores, 12 h; `CAMPAIGNS`
  colon-separated through `qsub -v`; Block 2 with `--sample 0`; output
  `refractory_audit_full_<UTC>/` in the submission folder.
- Test oracles: smoke test T3 (fake HOME; CODE clamped / not clamped /
  missing, LOGS table, DATA, VERDICT, unknown option refused).
- Confirmed: inferred.
- Status: smoke-tested 2026-10-30.

### Block 4: LindaColloDUP launch package

- Module: `hpc/Phenomenological_finalv1/LindaColloDUP/`: `launch_LCDUP.sh`,
  `submit_LCDUP.sh`, `apply_freeze_patch.py`, `smoke_test_freeze.py`,
  `apply_refractory_clamp.py`, `APPLY_LCDUP.txt` (procedure). Deployed on
  davinci in `Phenomenological/LindaColloDUP/`, with the pipeline copied
  from the cluster's 37-D `MEAMA/`, which is not in the repository.
- `launch_LCDUP.sh` settings (environment variable = default): MODE =
  Neuronal; SWEEP_GROUP = neuron_synapse; RHO_TOTAL = 1300; ASTRO_FRAC = 0;
  C_MAX = 350 um; SIMTIME = 180 s; FREEZE =
  Cm,I_inj,VA=-50,DeltaA,VR=-55,EC50_ampa,EC50_nmda,alpha_syn,O_G,Omega_G;
  CONN_RULE = flat; CONN_PERIODIC = 0; CONN_PROB_LO/HI = 0.1/0.6;
  STOA_GATE_LO/HI = 1.0/1.0; N_TOPOLOGIES_WORKER = 32; K_PARAMS = 1;
  WALLTIME = 24:00:00; TARGET = 1000000; HEADROOM_PCT = 200; CAMPAIGN_TAG =
  lcdup15_rho1300_neur_n01; CONDA_ENV = brian_final; DRYRUN = 0.
- Guards of `launch_LCDUP.sh`: exit 2 missing pipeline or package files;
  exit 3 SIZING_CONFIRMED not set, or REFRACTORY_CLAMP unset or not yes/no;
  exit 4 REFRACTORY_CLAMP disagreeing with the file, MODE, ASTRO_FRAC, gate,
  conn_prob bounds, FREEZE syntax, existing campaign tag; exit 5 seed
  allocator missing.
- `apply_freeze_patch.py`: adds `--freeze` to the folder's own
  `HPC_main_sweep.py` (3 insertions, marker `[LindaColloDUP patch 2026-10-29]`);
  `--check`, `--self-test`; refuses CR bytes, partial or foreign patches,
  missing or duplicate anchors.
- `apply_refractory_clamp.py`: Block 1's flag on the folder's own
  `ASD_fun_BD_cpp.py` (marker `[LindaColloDUP clamp 2026-10-29]`); `--check`,
  `--self-test`; exit 0 patched or already clamped, 1 refused.
- Test oracles: `smoke_test_freeze.py` (35 checks); the two patchers'
  `--self-test` (the clamp's includes a Brian2 parse of the patched
  equations); the dry run (Nn = 159, two qsub lines `-J 0-651%2`); the worker
  probe of `APPLY_LCDUP.txt` step 6 (NaN warnings 0 when clamped).
- Data flow: launches campaigns from the patched folder copies.
- Confirmed: yes 2026-10-29 for the settings the author decided (freeze
  list, SIMTIME, RHO_TOTAL, the rest as MEAMA) and 2026-10-30 for
  REFRACTORY_CLAMP = yes; inferred for VA = -50 mV and VR = -55 mV, and for
  I_inj = 4.0 pA (both open, section 8).
- Status: smoke-tested 2026-10-29 in the sandbox on 36-D copies (skipped:
  the 37-D files and davinci).

## 6. Pipeline entry points

| Command | Purpose | Minimal configuration for testing | Expected runtime |
|---|---|---|---|
| `python smoke_test_refractory_scan.py` | Blocks 2 and 3 | none (synthetic campaigns) | seconds |
| `bash check_refractory_nan.sh --sample 300` | audit of existing campaigns | a campaign folder with `job_args.json` and `topo_*/iter_*.npz` | minutes (not timed on davinci) |
| `qsub -q cpu submit_refractory_scan.sh` | per-run tables for every run | -- | up to 12 h |
| `python apply_refractory_clamp.py --self-test` | Block 4 clamp patcher | embedded fixture | seconds |
| `python apply_freeze_patch.py --self-test` | Block 4 freeze patcher | embedded fixture | seconds |
| `SIZING_CONFIRMED=1 REFRACTORY_CLAMP=yes DRYRUN=1 bash launch_LCDUP.sh` | Block 4 dry run | the LindaColloDUP folder on davinci | seconds |

## 7. Not executable in the sandbox

- Anything on davinci: PBS submission, the 37-D pipeline files (`MEAMA/`,
  `LindaColloDUP/`), real campaign sizes, job-log naming, read speed.
  Substitutes used: 36-D `main` copies with no-op parser options for the
  37-D flags; a stub `qsub`; campaign trees built from real worker output.
- Runs at the campaigns' length (180-200 s); the checks used 5 s, and 30 s
  for false positives.

## 8. Open questions

- Confirm Block 2's thresholds (t_early = 1 s; the late rule) (raised 2026-10-30).
- LindaColloDUP: the VA/VR option and I_inj (raised 2026-10-29).
- MEAMA: a clamped campaign in a folder of its own, or not (raised 2026-10-30).
- Where this project's decision log lives (raised 2026-10-29).
- Astrocyte variables ending NaN in Full runs at nominal parameters: a
  separate failure, not investigated (raised 2026-10-29).
