# SPEC -- Astro-Neuron-Network (hpc/): refractory hold, refractory audit, LindaColloDUP and GiuliaGate launch packages

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

Last reviewed at commit: 4aa3fdb (first draft at 562a9fd; Block 5, the repository
invariant of Block 1 and section 5.1 added on 2026-10-30 on top of 5f21a3d;
Block 5's REFUSED rule for an unrecognised CAdEx line added on top of 4aa3fdb;
its GIT CLONE status and queued-job hold added on top of 4e12a40; Block 4's
g box patch, conn_prob box, VA/VR values and clone folder, and Block 6's
two-node split, added on top of 15f82c9; Block 7 and Block 1's GitHub-wide
check added on top of a811543; Block 6 rewritten as the standard prior range
calibration on top of dcb52ab; Block 9, Block 6's four-node split, GiuliaGate
setup and astrocyte outcomes, and the diagnosis of the astrocyte NaN
(section 8) added on top of fbc20f6; Block 10, the mean astrocyte drive,
and GiuliaGate's O_N box added on top of 04dcaac)

## 0. Coverage

- `hpc/refractory_audit/` (the audit and its `checks/`)
- `hpc/Phenomenological_finalv1/LindaColloDUP/` (launch package and its `checks/`)
- `hpc/Phenomenological_finalv1/GiuliaGate/` (launch package of Giulia's new
  campaign, Block 9)
- the astrocyte's glutamate drive (Block 10): the coupling line of
  `Synapse_to_astro` in `hpc/Phenomenological_finalv1/ASD_fun_BD_cpp.py` and
  `smoke_test_astro_drive.py` -- not the rest of the astrocyte model
- `hpc/range_check/` (the standard prior range calibration, Block 6)
- the seed guard (Block 7): `hpc/Phenomenological_finalv1/seed_alloc.py`, its
  identical copies in `MEAMA/` and `LindaColloDUP/`, its smoke test
  `smoke_test_seed_allocation.py`, and the seed-allocation blocks of
  `launch_campaign.sh`, `MEAMA/launch_MEAMA.sh` and
  `LindaColloDUP/launch_LCDUP.sh` -- not the rest of those launchers
- the refractory handling of every CAdEx neuron model in the repository
  (today `hpc/Phenomenological_finalv1/ASD_fun_BD_cpp.py` and
  `hpc/Phenomenological_finalv1/Giulia/ASD_fun_BD_cpp.py`): in
  `Neuronal_Network`, the dV/dt line and the threshold, reset and refractory
  arguments of the neuron `NeuronGroup` -- not the rest of those files
- the root pipeline's 37-D registry, sweep groups and astrocyte contact gate
  (Block 8): `hpc/Phenomenological_finalv1/HPC_main_sweep.py` and
  `HPC_single_run.py` as far as Block 8 describes them, and the `O_N` lines
  of `ASD_fun_BD_cpp.py` -- not yet the model's equations or the rest of the
  sweep loop (their entries are being written with the author, 2026-10-31)

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
    cluster's `MEAMA/` (its campaigns run there). Superseded for `MEAMA/` by
    the next entry: MEAMA is rerun clamped, and Block 5 patches its folder
    once its arrays are stopped.
  - 2026-10-30 [author]: every project is rerun with the clamped model --
    Giulia's simulations, MEAMA, LindaColloDUP and the rho1300 family (the
    100-electrode project, confirmed by the author on 2026-10-30) -- so every
    project folder must hold a clamped `ASD_fun_BD_cpp.py` before its rerun
    (section 5.1, Block 5). Reruns use new campaign tags; data from the
    unclamped model is never pooled with clamped runs.
  - 2026-10-30 [author]: the campaigns run with the unclamped model are kept
    apart from the clamped reruns and deleted later; where they are is
    recorded in the project doc `claude/Unclamped_Campaigns_Inventory.md`.
    Clamped tags carry `clamp` in their stem, and any tool that reads several
    campaigns at once is given the clamped campaigns explicitly (for
    `multi_campaign_report.py`, `--glob` with the clamped stem; its default
    `campaign_*` takes every campaign in the folder).
  - 2026-10-30 [author]: the rho1300 rerun is Neuronal, as the eight
    `campaign_cadex_rho1300v*` (637 neurons, no astrocytes).
  - 2026-10-30 [author]: before relaunching, the activity produced over the
    prior boxes is measured with 20 s runs (Block 6) -- now for Giulia's
    setup and LindaColloDUP's, each as an independent 48-core job; the boxes
    of MEAMA and of the rho1300 family are retuned later (section 8).
    [corrected 2026-10-30] Each setup now runs on two 48-core nodes (the
    range-check entry below).
  - 2026-10-30 [author]: every running campaign is stopped, and the reruns
    are launched from folders of the git clone on davinci,
    `/davinci-1/home/ldellamea/ANN/Astro-Neuron-Network`: everything under
    the clone is clamped, every other `campaign_*` folder under `ANN/` is
    old. Tags as proposed: `cadex_rho1300_clamp_v1`, `cadex_hhgap_clamp_v1`,
    `meama_rho1600_clamp_n01`; LindaColloDUP keeps
    `lcdup15_rho1300_neur_n01` (never launched, and launched only clamped;
    the one exception to the `clamp` stem). LindaColloDUP goes first, from
    `hpc/Phenomenological_finalv1/LindaColloDUP/` in the clone; its 37-D
    pipeline is copied in from the cluster's `MEAMA/` and kept out of git
    (`.gitignore`, as is the campaign output) until the author sends the
    files to be versioned.
  - 2026-10-30 [author]: LindaColloDUP values. VA = -48 mV and VR = -58.2 mV,
    the registry nominals (the cluster's `MEAMA/` registry has the root
    copy's neuron and synapse rows: the author's paste, 2026-10-30);
    I_inj = 4.0 pA; conn_prob ~ U[0.01, 0.5] per topology, to lighten the
    simulations, with the density (1300 neurons/mm^2) and the culture width
    (C_MAX 350 um, so Nn = 159) unchanged; g_ampa and g_nmda boxes
    [0.01, 10] nS, log-uniform, nominal 0.3 nS (registry box [0.01, 5] nS).
  - 2026-10-30 [author]: the range check of a setup runs on two 48-core
    nodes, split as proposed by Claude: the one-axis sweeps on one node, the
    prior sample (16 topologies) on the other, and the analysis as a short
    job once both have ended (Block 6). [corrected 2026-10-30] Superseded by
    the next entry.
  - 2026-10-30 [author]: the standard prior range calibration, for every
    setup: the corners of the prior box (every free axis and conn_prob at
    their minimum or maximum, all combinations; a two-level fraction keeping
    every main effect and two-axis interaction when the 2^d corners exceed
    16,384) and as many random cells of the 7-bin grid (bin midpoints), as
    two 48-core jobs with a 24 h walltime, then the analysis job (Block 6).
    It replaces the one-axis sweeps and the prior sample.
  - 2026-10-30 [author]: the seed guard covers every launch folder under the
    cluster's ANN folder, not only the launching one (Block 7): the reruns
    start in fresh folders of the git clone, which hold no seed history.
    Asked by the author ("double check the seed guard") under the standing
    rule of 2026-08 that every campaign launch gets seeds no other launch has
    used.
  - 2026-10-30 [author]: every CAdEx model on GitHub carries the refractory
    hold, the second branch `Inverse-Transform-Sampling` included (Block 1).
  - 2026-10-31 [author]: the 37-D pipeline with the astrocyte contact gate,
    as the cluster's `MEAMA/` runs it, is the repository default at
    `hpc/Phenomenological_finalv1/` (Block 8), with section 2's hold; the
    36-D root pipeline is retired (last on `main` at c06363b). The
    independent tester runs on this version once the model's equations are
    specified and confirmed, as a night run (decided 2026-10-31).
  - 2026-10-31 [author]: every campaign runs from the git clone on davinci,
    so what runs is in the repository ("we will run from the cloned repo in
    the cluster thus what i run is in the repo"). Nothing is copied in from
    a cluster folder outside the clone: LindaColloDUP's step 0 copies the
    pipeline from the clone's root folder (Block 4), and Block 6's giulia
    setup moves from `Giulia_Astro/` to the new Giulia folder of the clone
    when that is built. The cluster's other copies are no longer checked.
  - 2026-10-31 [author]: Giulia's new campaign (GiuliaGate, Block 9) runs
    from its own folder of the clone, `hpc/Phenomenological_finalv1/GiuliaGate/`,
    with the root pipeline copied in ("They have their own copied and
    updated version in the clone"), MODE Full. Its prior, answering the open
    points of section 8: the default cell, every neuron axis at its registry
    value except Sigma, free in [2, 10] mV; C_Theta fixed; O_beta, O_3K and
    Omega_5P fixed; the asynchronous release fixed (U_0_ar, U_max,
    Omega_f_ar); g_ampa and g_nmda in [0.05, 10] nS; conn_prob in [0.05,
    0.4]; the contact gate drawn per network from U[0.2, 1] ("it's the
    standard model"; "It is the same as in MEAMA"); 180 s per simulation;
    the densities as proposed, 1200 neurons and 1200 astrocytes per mm^2 on
    300 um ("Densities are ok"). The other axes of Giulia's group
    `synapse_astro` are drawn from the registry boxes as proposed: U_0_sr,
    Omega_f_sr, Omega_d, alpha_syn, x0, O_G, Omega_G, I_bias, F, U_A, G_T,
    O_N. 15 free axes in all. This closes "when that is built" in the
    clone entry above: Block 6's giulia setup is this folder.
  - 2026-10-31 [author]: GiuliaGate's range check runs on four 48-core
    jobs ("Can you run the range check over 4 jobs with 48 cores?") and
    reports astrocyte outcomes ("Add astrocyte outcomes"); the gate is drawn
    per network, not designed (Block 6).
  - 2026-10-31 [author]: the astrocyte NaN of section 8 is fixed by option
    (a), "use the option a": an astrocyte's glutamate drive is the mean of
    Y_A over its links, not the sum (Block 10), in the root model, so for
    every Full campaign from now on. Option (b), the exact per-step update
    of Gamma_A, is not adopted.
  - 2026-10-31 [author]: GiuliaGate's O_N box is [0.0001, 1] per (uM s),
    log-uniform. Asked first for the upper bound, the author chose 2 (the
    recommended cap, inside RK4's stability bound under the mean drive);
    asked for the lower bound, the author answered "do also 0.0001 up to 1,
    the gate p is useful for the number of synapses influenced", read here
    as the box [0.0001, 1]. The gate stays: under the mean drive p sets how
    many synapses are linked, in both directions (glutamate into the
    astrocyte's mean, gliotransmitter back onto synapses), and how many
    astrocytes have any link; it does not rescale the drive of a linked
    astrocyte (Block 10).
  - 2026-10-29 [author]: LindaColloDUP runs MODE Neuronal; freezes Cm,
    I_inj, VR, VA, DeltaA, EC50_ampa, EC50_nmda (and the inert alpha_syn,
    O_G, Omega_G); SIMTIME 180 s; RHO_TOTAL 1300 cells/mm^2; everything else
    as MEAMA. The frozen values of VA, VR and I_inj were decided on
    2026-10-30 (entry above).

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
- Repository invariant: in every model file (a file named
  `ASD_fun_BD_cpp.py`, the module the workers import), the dV/dt line that
  contains the CAdEx term `exp((V-VT)/DeltaT)` ends with ` (unless refractory)`,
  and no Hodgkin-Huxley model file carries the flag (`hpc/ASD_fun_BD_cpp.py`
  must stay unflagged: holding V would break an HH neuron). Standalone checks
  that mirror the model use the clamped equation by default
  (`LindaColloDUP/checks/check_rs_reset_adaptation.py`; `--unclamped` gives
  the old model). Test fixtures that deliberately hold the unclamped line are
  exempt: `ASD_LINE` in `smoke_test_refractory_scan.py`, `_FIXTURE` in
  `apply_refractory_clamp.py`, `_CADEX` in `clamp_cadex_copies.py`. Oracle:
  `python3 hpc/refractory_audit/clamp_cadex_copies.py <repository root>` exits
  0 and lists exactly the two files above as CLAMPED and
  `hpc/ASD_fun_BD_cpp.py` as SKIP.
- GitHub-wide check (2026-10-30, asked by the author): every voltage
  equation in every tracked file of both branches was read (`git grep` for
  `d v.../dt =` and for the exponential term in any spelling). `main`
  a811543: the two files above are the only runnable CAdEx models, both
  clamped; the other hits are Hodgkin-Huxley models (top-level legacy
  scripts, `hpc/ASD_fun_BD_cpp.py`), a toy LIF in `MEAMA/env_check.py`, test
  fixtures, detection code and documentation. Branch
  `Inverse-Transform-Sampling`: its CAdEx copy
  `hpc/Phenomenological main/ASD_fun_BD_cpp.py` (an older model, `I` in place
  of `I_inj*bias_unit`) got the same flag and comment lines in 2f5b260;
  Brian2 parses it, `clamp_cadex_copies.py` rates it CLAMPED, and its own
  sweep pipeline on identical draws (60 neurons, 3 s, 6 runs, seed 1000)
  logged 4 NaN warnings unclamped and 0 clamped. (That commit's message says
  the branch no longer holds its sweep pipeline; it does.) Not CAdEx and not
  changed: the Izhikevich model of the top-level `ASN_fun_pheno.py` (used
  only by `Main_code_pheno.py`) has `refractory=2*ms` and no hold, the same
  failure class (section 8).
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
  `submit_LCDUP.sh`, `apply_freeze_patch.py`, `apply_box_patch.py`,
  `smoke_test_freeze.py`, `apply_refractory_clamp.py`, `APPLY_LCDUP.txt`
  (procedure), `.gitignore`, `seed_alloc.py` (Block 7; tracked here since
  2026-10-30, not copied from `MEAMA/`, whose copy cannot scan other
  folders). Deployed on davinci in this same folder of the git clone
  (decided 2026-10-30, section 4), with the 7 pipeline files copied in from
  the cluster's 37-D `MEAMA/`, which is not in the repository [corrected
  2026-10-31: copied from the clone's root folder `hpc/Phenomenological_finalv1/`
  (Block 8; section 4, run from the clone): `HPC_main_sweep.py` and
  `HPC_single_run.py` equal `MEAMA/`'s, `ASD_fun_BD_cpp.py` is `MEAMA/`'s with
  the hold]. `.gitignore` (anchored patterns) keeps the copies, their
  backups (`*.bak-*`, `*.crlf_bak`), `campaign_*/` and `artifacts/` out of
  git; `git clean -x` would delete them.
- `launch_LCDUP.sh` settings (environment variable = default): MODE =
  Neuronal; SWEEP_GROUP = neuron_synapse; RHO_TOTAL = 1300; ASTRO_FRAC = 0;
  C_MAX = 350 um; SIMTIME = 180 s; FREEZE =
  Cm,I_inj,VA=-48,DeltaA,VR=-58.2,EC50_ampa,EC50_nmda,alpha_syn,O_G,Omega_G
  (VA and VR written out at the registry nominals, so a changed registry
  cannot move them); CONN_RULE = flat; CONN_PERIODIC = 0; CONN_PROB_LO/HI =
  0.01/0.5; STOA_GATE_LO/HI = 1.0/1.0; N_TOPOLOGIES_WORKER = 32; K_PARAMS = 1;
  WALLTIME = 24:00:00; TARGET = 1000000; HEADROOM_PCT = 200; CAMPAIGN_TAG =
  lcdup15_rho1300_neur_n01; CONDA_ENV = brian_final; DRYRUN = 0. The g boxes
  are not launcher settings: they live in the folder's `HPC_main_sweep.py`
  after `apply_box_patch.py`; the launcher prints them (`g boxes : ...`) and
  writes them into the PROVENANCE file.
- Guards of `launch_LCDUP.sh`: exit 2 missing pipeline or package files, a
  `HPC_main_sweep.py` without the freeze patch, or one without both box
  markers; exit 3 SIZING_CONFIRMED not set, or REFRACTORY_CLAMP unset or not
  yes/no; exit 4 REFRACTORY_CLAMP disagreeing with the file, MODE,
  ASTRO_FRAC, gate, conn_prob bounds, FREEZE syntax, existing campaign tag;
  exit 5 seed allocator missing, an allocator without `--scan-tree`, a
  missing SEED_SCAN_TREE, or no free seed range (Block 7). `submit_LCDUP.sh`
  repeats the freeze and box guards on the compute node (exit 2).
- `apply_freeze_patch.py`: adds `--freeze` to the folder's own
  `HPC_main_sweep.py` (3 insertions, marker `[LindaColloDUP patch 2026-10-29]`);
  `--check`, `--self-test`; refuses CR bytes, partial or foreign patches,
  missing or duplicate anchors.
- `apply_box_patch.py` (decided 2026-10-30): in the folder's own
  `HPC_main_sweep.py`, rewrites the two rows of the `PARAM_BOUNDS` array
  whose comments name g_ampa and g_nmda from the registry box (0.01, 5.0) to
  (0.01, 10.0) nS and appends the marker `[LindaColloDUP box 2026-10-30]` to
  each; nothing else in the file changes. Everything derived from
  `PARAM_BOUNDS` follows at import: both axes stay in `LOG_PARAMS` (both
  bounds > 0 and log10(hi/lo) >= 1), the theta box becomes
  [ln 0.01, ln 10], and `job_args.json` `_axis_declaration` records
  low 0.01, high 10, log true. Refuses, writing nothing (exit 1): CR bytes,
  not UTF-8, no `PARAM_BOUNDS` block, a row missing or found twice, a box
  other than (0.01, 5.0) or (0.01, 10.0), the new box without the marker, a
  partial patch. Exit 0 patched or already patched. Backup
  `HPC_main_sweep.py.bak-box-<YYYYmmdd_HHMMSS>`; atomic write keeping the
  permissions; re-read (must classify as patched) and `py_compile`; restores
  the backup on failure. `--check`, `--self-test`, `--file`.
- `apply_refractory_clamp.py`: Block 1's flag on the folder's own
  `ASD_fun_BD_cpp.py` (marker `[LindaColloDUP clamp 2026-10-29]`); `--check`,
  `--self-test`; exit 0 patched or already clamped, 1 refused.
- Custom code: the two patchers and the box rule; no library edits a
  registry in place.
- Environment of `submit_LCDUP.sh` (2026-10-31): after `conda activate`
  the interpreter is `$CONDA_PREFIX/bin/python`, never a bare `python` from
  PATH (on davinci a module-loaded python can win on PATH after a successful
  activate; MEAMA's cluster job script resolves it the same way); exit 2
  when that file is not executable or fails the environment check.
- Test oracles: `smoke_test_freeze.py` (41 checks: T1-T9; T4 also sets a
  value off the nominal to prove NAME=VALUE is applied, since the campaign's
  values equal the nominals; T7 checks the g boxes in `job_args.json`; T8
  the box patch: both markers, box, log flag and theta box of each g axis;
  T9 the job script under stub module and conda: worker and manifest rebuild
  run with CONDA_PREFIX/bin/python, none with a python from PATH);
  the three patchers' `--self-test` (the box patcher's 20 checks include the
  refusals, the backup, idempotence and file permissions; the clamp's
  includes a Brian2 parse of the patched equations); the dry run (Nn = 159,
  `conn_prob : U[0.01, 0.5] per topology`, `g boxes : g_ampa [0.01, 10.0]
  nS; g_nmda [0.01, 10.0] nS`, two qsub lines `-J 0-651%2`); the worker
  probe of `APPLY_LCDUP.txt` step 6 (NaN warnings 0 when clamped; its
  `job_args.json` records 13 free axes, conn_prob 0.01-0.5, both g boxes,
  VA -48, VR -58.2, I_inj 4.0).
- Data flow: launches campaigns from the patched folder copies.
- Confirmed: yes 2026-10-29 for the settings the author decided (freeze
  list, SIMTIME, RHO_TOTAL, the rest as MEAMA); yes 2026-10-30 for
  REFRACTORY_CLAMP = yes, VA = -48 mV and VR = -58.2 mV, I_inj = 4.0 pA,
  conn_prob U[0.01, 0.5], the g boxes [0.01, 10] nS and the clone folder.
- Status: smoke-tested 2026-10-30 in the sandbox on a fresh folder built
  from the package files and a stand-in for the `MEAMA/` copy (the
  repository's root 36-D files, whose registry rows equal `MEAMA/`'s; the
  unclamped model of d0ced88; the three 37-D flags as no-op options):
  `APPLY_LCDUP.txt` steps 2-7 as written, the guards above, the probe.
  Skipped: the 37-D files and davinci. [2026-10-31] Re-run on a folder built
  from the cluster's own 37-D files (the author's zip of `MEAMA/`, Block 8):
  steps 2-7 as written: both patches CHECK OK and PATCHED, smoke test 41
  passed (with T9), CLAMPED, probe 2 ok with 0 NaN warnings and the gate
  declared inert, dry run with Nn=159. [2026-10-31] Step 0 from the clone's
  root folder (section 4), on a scratch copy of the clone: `git status`
  empty after the copy; both patches PATCHED; smoke test 41 passed; step 5a
  `ALREADY CLAMPED` with exit 0 twice; probe 2 ok with the clamped dV/dt
  line and 0 NaN warnings; dry run with Nn=159, the freeze list and
  `V clamp (t_ref): yes`; Block 6's precheck OK on the folder.

### Block 5: clamp_cadex_copies, the cluster-wide clamp

- Module: `hpc/refractory_audit/clamp_cadex_copies.py` (standard library only;
  Brian2 only in its self-test, optional).
- CLI: `python3 clamp_cadex_copies.py [ROOT ...] [--apply] [--force-running]
  [--recent-hours H] [--max-depth D] [--self-test]`; default ROOT
  `/davinci-1/home/ldellamea/ANN`, H = 6 h, D = 6.
- Finds: every file named `ASD_fun_BD_cpp.py` under the roots (the module
  name every worker imports from its `lib_dir`), descending at most D levels
  and skipping folders named `campaign_*`, `scratch*`, `mea_out*`, hidden
  folders and `__pycache__`.
- Classifies each file: SKIP (the CAdEx term `exp((V-VT)/DeltaT)`, spaces
  ignored, appears nowhere in the file), CLAMPED (one CAdEx dV/dt line ending
  in the flag), NOT CLAMPED (one CAdEx dV/dt line without flags), REFUSED (CR
  bytes, not UTF-8, more than one CAdEx dV/dt line, other flags on it, the
  marker `[refractory clamp]` present without the flag, or the term present
  but carried by no single-line `dV/dt = ... : volt`). An unrecognised CAdEx
  copy is therefore REFUSED, never SKIP.
- A NOT CLAMPED file tracked by a git clone (git `ls-files --error-unmatch`
  succeeds in its folder; without git, a `.git` entry in its folder or above)
  becomes GIT CLONE, in both modes, and is never edited: a local edit would
  block every later `git pull` there; the remedy is `git pull` in that clone.
- Queued jobs: `qstat -u $USER`, then `qstat -f <id>` per job (lines rejoined
  where qstat wraps them with newline + TAB); a job belongs to a folder when
  its `Variable_List` has `PBS_O_WORKDIR` or `REPO_DIR` equal to that folder
  (real paths). Report mode names such jobs in the file's row. Without
  qstat, or for a job whose record names neither variable, a WARNING line is
  printed and the verdict line points to it; the check then does not hold
  anything (fail open, loudly).
- With `--apply`, for each NOT CLAMPED file: HELD (left untouched) if a job
  of the user still listed by qstat belongs to its folder, or if a
  `campaign_*` folder next to it, or one of that folder's children, was
  modified less than H hours ago, and `--force-running` is not given;
  otherwise backup `<file>.bak-clamp-<YYYYmmdd_HHMMSS>`, insert one comment
  line holding the marker above the dV/dt line, append ` (unless refractory)`,
  write atomically keeping the permissions, re-read, re-classify (must be
  CLAMPED), syntax-check; on any failure restore the backup and report
  FAILED; else PATCHED. Nothing else in the file changes.
- Output: one row per file (status, age of the newest campaign output in its
  folder, path, the dV/dt line or the reason) and a verdict line.
- Exit status: 0 every CAdEx copy is clamped (after patching, with
  `--apply`); 3 report mode with at least one NOT CLAMPED copy; 1 any
  REFUSED, HELD, FAILED or GIT CLONE row, a missing root, or no CAdEx copy
  at all.
- Custom code: the scan and the patch rule; no library does this. Same patch
  rule as `hpc/Phenomenological_finalv1/LindaColloDUP/apply_refractory_clamp.py`,
  with the CAdEx-term condition added and a neutral marker.
- Test oracles: `--self-test` (28 checks: classification of each case, the
  patch adds only the comment line and the flag, HH untouched, HELD and
  `--force-running`, backups, idempotence, exit codes, a copy tracked by a
  git clone, qstat parsing with a wrapped `Variable_List` and a job without
  one, HELD by a queued job, no qstat, Brian2 parse of the patched
  equations); `smoke_test_clamp_cadex_copies.py` (18 checks: a mock of the
  cluster folders built from the repository's real model files, report exit
  3, apply exit 0, byte-exact diff, one backup per patched file, second apply
  a no-op; bytes; a stub `qstat` on PATH: a queued job holds its folder, none
  left lets the patch through).
- Confirmed: yes 2026-10-30 for the goal (every project folder runs the
  clamped model, section 4); inferred for the H = 6 h hold rule, the pruned
  folder names, the git-clone rule and the queued-job rule (the real
  `qstat -f` layout on davinci has not been read).
- Status: smoke-tested 2026-10-30 in the sandbox (skipped: davinci).

### Block 6: range_check, the standard prior range calibration

- Purpose (decided by the author on 2026-10-30, the standard for every
  setup): before a campaign is launched, measure the activity its prior box
  produces, through the setup's own sweep code, at the corners of the box and
  at random cells of its B-bin grid; 20 s per run.
- Modules (`hpc/range_check/`): `range_design.py` (the designs; pure
  functions, numpy only), `range_check.py` (driver), `range_analysis.py`
  (metrics, labels, summaries), `range_plots.py` (figures only),
  `submit_range_check.sh` (the PBS job), `launch_range_check.sh` (login node:
  one setup as three PBS jobs on two 48-core nodes, or five on four with
  NODES=4),
  `smoke_test_range_check.py`, `README_RANGE_CHECK.txt`.
- Notation. For a setup with sweep group G and freeze list Phi:
  F = (j_1, ..., j_dr), the indices of G minus Phi (`resolve_sweep_group`,
  the module's `_apply_freeze`) whose theta box has lo_j < hi_j, is the set of
  free axes; theta_j is the sweep's coordinate of axis j (`PARAM_BOUNDS_THETA`:
  ln on `LOG_PARAMS` axes, natural units otherwise), box [lo_j, hi_j].
  conn_prob is a design dimension when the setup's conn_rule is flat and its
  box [c_lo, c_hi] (`--conn_prob_lo/hi`) has c_lo < c_hi, in natural units.
  d = dr + 1 with conn_prob, d = dr without. theta0 =
  natural_to_theta(NOMINAL_PARAMS) after the freeze. N_w = n_workers (the
  module's `_resolve_n_workers`). seed = `--rc-seed` (default 20261030).
- Corners: X in {-1, +1}^(N_c x d), columns F then conn_prob.
  - If 2^d <= N_max (`--rc-corner-budget`, default 16,384): the full
    factorial, N_c = 2^d, row r column k = +1 if bit k of r is 0, else -1.
  - Else a regular 2^(d-p) fraction: b = floor(log2 N_max), N_c = 2^b,
    p = d - b; columns 0..b-1 the full factorial of b; column b+i the product
    of the base columns set in the generator mask g_i. The masks come from the
    greedy search `range_design.fraction_generators` (random orders from
    `default_rng(seed + 20000)`), trying the resolution R = 8, 7, 6, 5 in turn
    (8 orders each) and keeping the first found; `min_word_length` verifies
    that every word of the defining relation, |G| + popcount(xor of the masks
    in G) for a nonempty generator set G, is >= R. R >= 5: no main effect or
    two-axis interaction is aliased with another. None found: error.
  - A run with codes x: theta_j = lo_j where x_j = -1, hi_j where +1 (j in
    F), every other axis at theta0; conn_prob = c_lo where x_conn = -1, c_hi
    where +1.
  - Level L in {0, 1} (conn_prob at c_lo, c_hi; level 0 only when conn_prob
    is not a dimension): the n_L rows with x_conn = 2L - 1, permuted by
    `default_rng(seed + 20100 + L)`, split over T_c topologies
    (`--rc-corner-topologies`, default 8) of N_w k_c consecutive slots,
    k_c = ceil(n_L / (T_c N_w)); the T_c N_w k_c - n_L last slots hold the box
    centre, theta_j = (lo_j + hi_j) / 2 for j in F (centre points).
- Interior: B bins per dimension (`--rc-bins`, default 7) of equal width in
  the dimension's own coordinate, so equal prior mass under a uniform or
  log-uniform prior; midpoints m_(j,i) = lo_j + (i + 1/2)(hi_j - lo_j)/B,
  i = 0..B-1, and the same over [c_lo, c_hi] for conn_prob. Level L = the
  conn_prob bin, L = 0..B-1 (level 0 only when conn_prob is not a
  dimension): conn_prob = m_(conn,L); n = T_i N_w k_i cells,
  k_i = ceil(ceil(N_i / n_levels) / (T_i N_w)), N_i = `--rc-interior-runs`
  (default 16,384), T_i = `--rc-interior-topologies` (default 7); for each
  cell and each j in F independently, i_j ~ U{0, ..., B-1}
  (`default_rng(seed + 30000 + L)`), theta_j = m_(j,i_j); every other axis
  at theta0.
- Driver: `python3 range_check.py --rc-phase {corners,interior} --rc-level L
  --rc-out DIR [--rc-like JOB_ARGS] [--rc-bins B] [--rc-corner-budget N]
  [--rc-corner-topologies T] [--rc-interior-runs N] [--rc-interior-topologies
  T] [--rc-simtime T] [--rc-seed S] [--rc-subset N] [--rc-design-only]
  [--rc-allow-unclamped] <HPC_main_sweep.py arguments>`. It imports
  `HPC_main_sweep` from the `--lib_dir` among the pass-through arguments and
  calls its `main()` with them followed by `--out_dir DIR/corners_c<L>` or
  `DIR/interior_b<L>`, `--simtime` (default 20 s), `--n_topologies T`,
  `--n_params_per_worker k`, `--seed_master` (seed + 2000 + L for corners,
  seed + 3000 + L for the interior) and `--conn_prob_lo P --conn_prob_hi P`
  with P the level's conn_prob. Later arguments win. In the parent process
  only, `sample_theta` is replaced by a function that returns the padded
  design in order, so successive topologies take successive chunks of N_w k
  slots: the only change to the production path. `--rc-subset N`: the first
  N design points of the level, on one topology (tests). `--rc-like` rebuilds
  the base argument list from a campaign's `job_args.json` through the
  module's own parser (each key that is a parser destination; store_true
  options as flags when true; None values skipped; never out_dir,
  scratch_root, lib_dir, seed_master, n_workers, n_topologies,
  n_params_per_worker, simtime, rebuild_manifest_only); explicit arguments
  after it override. Refuses (exit non-zero, nothing run) an unclamped model
  unless `--rc-allow-unclamped`, an output folder that already holds runs,
  and a level outside the phase's range. Exit 1 when not every slot was drawn
  (a topology failed to build); the runs made stay usable.
- `design.json` (one per level folder): phase, level, free_axes, theta_lo,
  theta_hi, has_conn, conn_box, conn_prob, nominal_theta, n_workers,
  n_topologies, k, n_slots, n_design, n_padding; corners: corner_design (kind,
  n_runs, resolution, base, generators, d, n_levels, n_level_corners), codes
  (n_design x dr), centre_theta; interior: bins, n_levels, midpoints, cells
  (n x dr); and simtime, seed_master, lib_dir, module, model_line,
  param_names, log_params, argv.
- Analysis (`range_analysis.py DIR`): per run, the neuron spikes in the
  window [t_skip, T] (t_skip default 2 s, shifted to start at 0) go through
  `burst_metrics.compute_all` for the burst rate (log-ISI detector,
  `li_burst_rate_per_min`), CV of ISI and adaptation index; mean rate =
  spikes in the window / (Nn (T - t_skip)); active fraction = neurons with a
  window rate > 0.1 Hz. Labels come from the run's stored theta and its
  folder's design.json, independent of the draw order: corners, per free
  axis, code -1 or +1 where theta equals lo or hi and 0 at the box centre
  (tolerance 1e-9 of the box width), 'corner' when every code is +-1,
  'centre' when every code is 0; interior, per free axis the bin whose
  midpoint theta equals, 'interior' when every axis has one; else
  'unmatched' (exit 1). conn_prob: code 2L - 1 (corners), bin L (interior).
  Outcomes: y = log10(rate + 0.01 Hz); silent = 1[active fraction < 0.1];
  high = 1[rate > 20 Hz]; bursting = 1[burst rate > 0] (reporting
  conventions, not targets). Writes `runs.csv`; `corner_effects.csv`: for
  every dimension j, over the corner runs, E_j(f) = mean(f | x_j = +1) -
  mean(f | x_j = -1) for f = y, silent, high, bursting, with n(+) and n(-);
  `corner_interactions.csv`: for every pair (j, k), E_jk(y) =
  mean(y | x_j x_k = +1) - mean(y | x_j x_k = -1), the two-level interaction
  contrast; `interior_bins.csv`: per dimension and bin, n, rate quartiles,
  shares silent / high / bursting, median burst rate; `interior_pairs.csv`:
  per pair of dimensions and pair of bins, n, share silent, share high,
  median y; `interior_spearman.csv`: Spearman rho (`scipy.stats.spearmanr`)
  of each dimension (theta; the run's conn_prob) with y and with the burst
  rate; `calibration_summary.txt`: corner runs and the distinct corners of
  the design they cover, centre points, shares, the curvature mean y(centre)
  - mean y(corners), the main effects, the 20 largest interactions, the five
  lowest- and highest-rate corners, the interior bins where more than half
  the runs are silent or above 20 Hz, the Spearman list.
- Full mode (2026-10-31; a level folder whose `job_args.json` has mode Full
  and Na > 0, Na the density-resolved count): per run, from the astrocyte
  events `spk_A_t`, `spk_A_i` of the npz (threshold crossings C > C_osc of
  the astrocyte group) in the same window: astro rate = events in the window
  / (Na (T - t_skip)); astro active fraction = astrocytes with at least one
  event in the window; and the run's gate p (`stoa_gate_p` of the sidecar;
  NaN when absent). Outcomes y_astro = log10(astro rate + 0.001 Hz) and
  astro_silent = 1[astro active fraction < 0.1]. `corner_effects.csv` adds
  E_j(y_astro) and E_j(astro_silent); `interior_bins.csv` adds the astro
  rate quartiles and the share astro-silent; `interior_spearman.csv` adds
  Spearman rho of each dimension with y_astro; the summary adds the
  astrocyte shares, quartiles, curvature, main effects, astro-silent bins
  (more than half) and Spearman list. Runs without astrocyte metrics leave
  these NaN; a table without any (Neuronal setups) is unchanged.
  `gate_networks.csv` (only when some run carries a gate): one row per
  network (level folder, topology) with its p, conn_prob, number of runs,
  median y and, in Full mode, median y_astro and share astro-silent; dy =
  the network's median y minus the median of those over the networks of its
  folder (dy_astro likewise), which removes the conn_prob level a folder
  shares. The summary gives Spearman rho of p with dy and with dy_astro over
  the networks (at least 3 with spread, else "too few networks"). The gate
  is drawn per topology, not designed, so it is read across networks only.
- Plots (`range_plots.py DIR`): `corner_effects.png` (main effects on y and
  on the shares), `interior_bins.png` (one panel per dimension: median rate
  per bin with the interquartile band on one shared scale, shares per bin),
  `rate_distributions.png` (log rate of corners, centre points, interior).
  Full mode: a third panel of `corner_effects.png` (effects on y_astro and
  astro_silent), `interior_bins_astro.png` (median astro rate per bin with
  its interquartile band, share astro-silent), and a second panel of
  `rate_distributions.png` (y_astro).
- Job script (`submit_range_check.sh`, 48 cores, walltime 24 h, `-k eo`):
  SETUP = lcdup | giulia (else exit 2). lcdup: LIB_DIR default
  `/davinci-1/home/ldellamea/ANN/Astro-Neuron-Network/hpc/Phenomenological_finalv1/LindaColloDUP`
  (Block 4's folder); the settings of `launch_LCDUP.sh` written out (FREEZE
  default `Cm,I_inj,VA=-48,DeltaA,VR=-58.2,EC50_ampa,EC50_nmda,alpha_syn,O_G,Omega_G`
  or FREEZE_ENC decoded, `:` for `,` and `@` for `=`; `--conn_prob_lo 0.01
  --conn_prob_hi 0.5`; `--Nn 115` overridden by the density, so Nn = 159);
  refuses a LIB_DIR without both box markers. giulia (2026-10-31; until then
  `Phenomenological/Main/Giulia_Astro` with LIKE from its
  `campaign_cadex_hhgap_v5`): LIB_DIR default
  `/davinci-1/home/ldellamea/ANN/Astro-Neuron-Network/hpc/Phenomenological_finalv1/GiuliaGate`
  (Block 9), no LIKE. Before conda it refuses (exit 2) a LIB_DIR without
  `apply_prior_patch.py` or `prior_config.tsv`, or whose `HPC_main_sweep.py`
  lacks `def _apply_freeze(` or `--stoa_gate_lo`; with CHECK_ONLY=1 and a
  `python3` on PATH it also runs `apply_prior_patch.py --check-applied`
  (failure: exit 2). After the environment check, with
  `$CONDA_PREFIX/bin/python`: `--check-applied` (failure: exit 2), the
  freeze list from `--freeze-list` unless FREEZE_ENC is given, and the
  conn_prob and gate boxes from `--topology conn_prob` and `--topology
  stoa_gate_p` (each value must be a number, else exit 2; an empty freeze
  list: exit 2). Settings: `--mode Full --sweep_group tripartite --freeze
  <list> --density 1200 --density_astro 1200 --c_max 300 --Nn 115 --Na 115
  --conn_rule flat --conn_prob_lo/hi <TSV> --stoa_gate_lo/hi <TSV>
  --seed_stoa_gate 71`, the geometry and seeds as lcdup (`--dpi 100`).
  CORNER_LEVELS and INTERIOR_LEVELS may separate their levels with `:`
  (decoded to spaces); every level must be a non-negative integer (else
  exit 2). The banner prints the freeze list.
  EXTRA (space-separated) ends every call. PHASES = all (default) | corners |
  interior | none: corners at levels CORNER_LEVELS (default `0 1`) with
  `--rc-corner-topologies CORNER_TOPOS` (8) `--rc-corner-budget
  CORNER_BUDGET` (16384); interior at levels INTERIOR_LEVELS (default
  0..BINS-1, BINS 7) with `--rc-bins BINS --rc-interior-runs INTERIOR_RUNS`
  (16384) `--rc-interior-topologies INTERIOR_TOPOS` (7); ANALYZE = 1
  (default) | 0: `range_analysis.py OUT --workers <ncpus>` then
  `range_plots.py OUT`. Exit 2 before conda: PHASES or ANALYZE invalid,
  PHASES=none with ANALYZE=0, `range_check.py` or `range_design.py` missing
  from the submission folder, a LIB_DIR without `HPC_main_sweep.py`, the
  lcdup box guard, PHASES=none with no `OUT/*/design.json`. CHECK_ONLY=1
  runs those guards (all but the last), prints `[range_check] precheck OK:
  ...` and exits 0. Exit 2 also when `conda activate` or the environment
  check fails; otherwise 1 if any call failed, else 0. Every interpreter call
  after `conda activate` uses `$CONDA_PREFIX/bin/python` (2026-10-31; as
  `submit_LCDUP.sh`, Block 4): exit 2 when that file is not executable.
- Launcher (`launch_range_check.sh`, login node): refuses (exit 2) an
  unknown SETUP and a comma in OUT or in a passed variable (`qsub -v` splits
  on commas); runs the job script once with CHECK_ONLY=1 (failure: exit 2,
  nothing submitted); then submits into one OUT (default
  `~/ANN/range_check/<setup>_<UTC>`): job 1 `rc_corners` (Q_CORNERS, default
  intel, PHASES=corners ANALYZE=0), job 2 `rc_interior` (Q_INTERIOR, default
  cpu, PHASES=interior ANALYZE=0), both `select=1:ncpus=NCPUS` (48) and
  WALLTIME (24 h); job 3 `rc_analysis` (Q_ANALYSIS cpu, ANALYSIS_NCPUS 8,
  ANALYSIS_WALLTIME 2 h, `-W depend=afterany:<job 1>:<job 2>`, PHASES=none
  ANALYZE=1). Passed on when set: LIB_DIR, LIKE, BINS, CORNER_LEVELS,
  CORNER_TOPOS, CORNER_BUDGET, INTERIOR_LEVELS, INTERIOR_TOPOS,
  INTERIOR_RUNS, RC_SIMTIME, EXTRA, FREEZE_ENC, CONDA_ENV. A refused job 1:
  exit 3, nothing else submitted; job 2: exit 3 and the `qdel` line for job
  1; job 3: exit 3 and the manual command. DRYRUN=1 prints the three qsub
  lines and submits nothing. NODES = 2 (default, the above) | 4 (else exit
  2; 2026-10-31): four compute jobs, `rc_corners_a` and `rc_corners_b`
  (Q_CORNERS; PHASES=corners with CORNER_LEVELS=0 and 1), `rc_interior_a`
  and `rc_interior_b` (Q_INTERIOR; PHASES=interior with INTERIOR_LEVELS =
  0..H-1 and H..B-1 joined by `:`, B = BINS (default 7, at least 2, else
  exit 2), H = ceil(B/2)), then `rc_analysis` with `-W
  depend=afterany:<the four jobs>`; with NODES=4 a set CORNER_LEVELS or
  INTERIOR_LEVELS exits 2. A refused compute job k > 1: exit 3 and the
  `qdel` line for the jobs already submitted. DRYRUN=1 prints N + 1 qsub
  lines (N compute jobs).
- LindaColloDUP sizes: dr = 13, d = 14, so the full factorial: per conn
  level 8,192 corners + 256 centre points = 8 x 48 x 22 = 8,448 runs (16,896
  in all, conn_prob 0.01 and 0.5); interior 7 levels of 2,352 cells
  (7 x 48 x 7), 16,464 runs, conn_prob = 0.045 + 0.07 L.
- GiuliaGate sizes (Block 9): dr = 15, d = 16 > 14, so a 2^(16-2) fraction,
  N_c = 16,384, resolution 8 in the sandbox; per conn level 8,192 corners +
  256 centre points = 8 x 48 x 22 = 8,448 runs at conn_prob 0.05 and 0.4;
  interior 7 levels of 2,352 cells, conn_prob = 0.075 + 0.05 L. Gate p per
  topology: 16 corner networks and 49 interior networks.
- Custom code: the fraction's generator search (`range_design.py`
  docstring: pyDOE3 1.6.2's `fracfact_by_res(n, 5)` is not resolution V for
  n = 14 or 23, checked 2026-10-30), the labels and the effect contrasts.
- Test oracles (`smoke_test_range_check.py`, R1-R11 and R6b; 87 checks with
  Brian2, 76 and 2 skipped with `--no-sim`; 65 and 58 before 2026-10-31).
  Added 2026-10-31: R6b end to end in Full mode on a patched GiuliaGate copy
  (12 neurons, 6 astrocytes: Na = 6, astrocyte metrics and one gate p in its
  box in runs.csv, astrocyte columns in the corner effects,
  `gate_networks.csv`); R7 also plants astrocyte events (0.05 Hz and 0.4 Hz
  with B at its lower and upper bound, times p / 0.6 per network) and
  recovers E_B(y_astro) = 0.884 for a planted 0.892 with |E_A(y_astro)| =
  0.002, and the gate Spearman over 12 networks, 0.986; R9 the giulia setup
  (freeze list and both boxes from the TSV, Full, tripartite, 1200/1200 on
  300 um; `:` level lists; an unpatched copy and a missing TSV exit 2 before
  any `range_check.py` call); R10 NODES=4 (queues, levels, the dependency on
  all four, BINS=5, NODES=3, a level list with NODES=4, a refused third
  submission, the dry run, giulia on four nodes, an unpatched copy refused
  by the precheck); R11 the GiuliaGate design (the 15 free axes of the TSV,
  the 16-dimension fraction of resolution >= 5, 8,192 corners at conn_prob
  0.05, 2,352 interior cells at 0.075, Sigma linear [2, 10], g_ampa and
  g_nmda log [0.05, 10], every fixed axis at its TSV value). R1 the fraction for the 37-D root (the
  36-D root until 2026-10-31)
  pipeline (resolution 8 here, floor 7; no aliased main/two-axis pair in a
  Gram check; balanced; the two levels together are the design; every vector
  at the bounds its codes name; centre points; inactive axes at nominal); R2
  the full factorial with the LindaColloDUP freeze (each level all 2^13
  corners once; VA, VR held, also off the nominal); R3 `--rc-like`; R4 the
  37-D Giulia copy; R5 the interior (sizes, conn midpoints, every value a bin
  midpoint, chi-square uniformity over 91 tests, seed determinism); R6 end to
  end on a tiny network (labels equal the design's codes and bins, tables,
  figures, refusals); R7 a planted effect recovered (E_A = 0.5985 for a
  planted 0.6004; inert axes |E| < 0.003; interior medians doubling per bin;
  an off-grid run unmatched, exit 1); R8 bytes; R9 the job script and R10
  the launcher with stubs in a sealed environment, each probed first (in R9
  the stub conda activate points CONDA_PREFIX at a stub environment, and
  every interpreter call must go to its bin/python, none to a python on
  PATH). Ten
  planted faults (2026-10-30) were each caught by a named check.
- Confirmed: yes 2026-10-30 for the design (corners at every free axis's
  bounds in all combinations, the same number of random cells of the 7-bin
  grid, a fraction keeping every main effect and two-axis interaction when
  2^d > 16,384, two 48-core jobs with a 24 h walltime then the analysis job,
  the standard for every setup), the two setups and the 20 s runs; yes
  2026-10-31 for four 48-core jobs for GiuliaGate, the astrocyte outcomes,
  the gate drawn per network, and the giulia setup = GiuliaGate (section 4);
  inferred for T_c = 8, T_i = 7, the centre points, the seeds, t_skip = 2 s,
  the reporting cut-offs, the queue assignment, the four-node split (corner
  levels on intel, interior halves on cpu), the astrocyte definitions (rate
  floor 0.001 Hz, astro-silent below 0.1) and the per-network gate analysis.
- Status: smoke-tested 2026-10-30 in the sandbox (36-D root pipeline,
  freeze-patched copy, 37-D `Giulia/` copy; both phases with the exact
  LindaColloDUP settings on 159 neurons: corners at conn_prob 0.5, interior
  at 0.465, labels and analysis correct). Skipped: davinci, the cluster's
  gate version [2026-10-31: run on it, Block 8; 65 passed on the 37-D root].
  History: until 2026-10-30 (a3878d2) this block specified
  one-axis sweeps at three conn_prob levels and a prior sample, replaced by
  this design at the author's request. The R8 failure on davinci of
  2026-10-30 (an exported `module` function reaching the job) was fixed by
  the sealed environment, now used by R9 and R10; it passed on davinci at
  1757d3e. Smoke-tested 2026-10-31 in the sandbox after the four-node split,
  the giulia setup and the astrocyte outcomes: 87 passed (76 + 2 skipped
  with `--no-sim`). Skipped: davinci. The astrocyte outcomes of a real
  GiuliaGate run carry no information until section 8's astrocyte NaN is
  fixed (R6b checks the plumbing only). [corrected 2026-10-31] Fixed by
  Block 10; 87 passed again with it.

### Block 7: the seed guard across ~/ANN

- Module: `hpc/Phenomenological_finalv1/seed_alloc.py` (standard library
  only), byte-identical copies in `MEAMA/` and `LindaColloDUP/`; callers: the
  seed blocks of `launch_campaign.sh` (rho1300 family), `MEAMA/launch_MEAMA.sh`
  and `LindaColloDUP/launch_LCDUP.sh`.
- Invariant (author, 2026-08, extended 2026-10-30): a launch's seed span
  [base, base + span) contains no `_resolved_seed_master` recorded by any
  campaign and overlaps no span reserved by any launch, in its own folder and
  in every launch folder under SEED_SCAN_TREE (default
  `/davinci-1/home/ldellamea/ANN`).
- `scan_tree(tree, max_depth=6)`: `os.walk` of `tree`, symbolic links to
  folders not followed. In every folder at most `max_depth` levels below
  `tree` (`tree` is level 0): the child `campaign_*` folders are read as one
  launch folder (`used_seeds_from_job_args`: `campaign_*/sweep_*/job_args.json`,
  truncated files counted and skipped) and not entered; `artifacts/seed_ledger.tsv`,
  if present, is read as a ledger. Not entered: hidden folders, names starting
  `campaign_`, `scratch`, `mea_out`, `topo_`, and `__pycache__`, `artifacts`.
  Returns the seeds, the spans and counts (campaign folders, readable and
  unreadable job_args.json, ledgers, unreadable ledgers and folders). A tree
  that is not a directory raises ValueError.
- `allocate(..., scan_trees, scan_depth)`: the own folder's seeds and ledger
  (unchanged) plus every scan; spans read twice (the own ledger lies inside
  the tree) are de-duplicated; a random base is redrawn until the span is
  free (256 attempts, else RuntimeError, exit 3). Only the own `--ledger` is
  written.
- CLI: `--scan-tree DIR` (repeatable), `--scan-depth N` (default 6); a
  missing tree exits 2 with no base on stdout; `--report` prints each scan
  (campaign folders, job_args.json, seeds, ledgers with their span counts)
  and the totals. Without `--scan-tree` the behaviour is unchanged.
- Launchers: `SEED_SCAN_TREE` (default the ANN folder) passed as
  `--scan-tree`; refuse (exit 5) an allocator without `--scan-tree`, a
  missing tree and a failed allocation, with nothing submitted; the header
  prints `Seed guard : this folder and every launch folder under ...`;
  `launch_LCDUP.sh` also writes it, and the sha256 of `seed_alloc.py`, into
  the PROVENANCE file. A dry run still reserves its span (note
  `DRYRUN-not-submitted`). `launch_campaign.sh` and `launch_MEAMA.sh` keep
  their existing fallback when no allocator is found (raw OS entropy with a
  warning); the LindaColloDUP launcher refuses.
- Not covered by the scan, by design: the range check's runs (fixed seeds,
  outside any `campaign_*` folder, never training data); launchers outside
  the three above (`Burst Tests/`, `Multiple Sweeps/` with its hard-coded
  bases, the MEAMA benchmark's fixed seed).
- Test oracles: `smoke_test_seed_allocation.py` T7 (10 checks): the scan
  reads the campaigns and ledgers of launch folders at depths 1 and 2 and
  skips hidden, `scratch*`, `topo_*`, inside-campaign and too-deep folders;
  `--scan-depth 7` reaches depth 7; a missing tree is refused; from one
  folder, 40 allocations in a small seed space overlap a neighbouring
  folder's span without the scan (16 of 40 in the run of 2026-10-30) and
  never with it; only the own ledger is written; `--report` lists the scan.
  Groups T1-T6 unchanged (47 checks in all).
- Confirmed: yes 2026-10-30 for the scope (all of ANN) and the refusal of a
  missing tree; inferred for the depth 6, the skipped folder names and the
  default path.
- Status: smoke-tested 2026-10-30 in the sandbox; the three launchers run
  against a mock ANN tree (an old MEAMA-like campaign and ledger beside the
  clone): each allocated clear of it and refused (exit 5, no qsub call) an
  old allocator; `launch_LCDUP.sh` also refused a missing tree and a tree
  whose ledgers reserve the whole seed space. A git pull replaces an ignored
  copy of MEAMA's allocator left by an earlier step 0 (git 2.43). Skipped:
  davinci (the real ANN tree, its size and its git 2.31).

### Block 8: the 37-D pipeline with the astrocyte contact gate (the default)

- Module: `hpc/Phenomenological_finalv1/HPC_main_sweep.py`,
  `HPC_single_run.py`, `ASD_fun_BD_cpp.py`, `synapse_pdist.csv`: the
  pipeline the cluster's `Phenomenological/MEAMA/` runs, the repository
  default since 2026-10-31 (decision in section 4). Callers: every launcher
  that uses the root folder or a copy of it (`launch_campaign.sh` /
  `submit_sweep_mixed.sh`, `MEAMA/`, `LindaColloDUP/`, Block 6).
- Provenance (the author's zip of `MEAMA/`, 2026-10-31): `HPC_main_sweep.py`
  (sha256 d609504e...) and `HPC_single_run.py` (f71851d5...) byte-identical
  to the cluster's; `ASD_fun_BD_cpp.py` = the cluster's file (01e4f998...,
  unclamped) plus Block 1's hold, so it differs from the cluster's only by
  section 2's `(unless refractory)` and its four comment lines;
  `smoke_test_param_recording.py` (ebcdb93b...) byte-identical;
  `smoke_test_stoa_gate.py` was already identical. `synapse_pdist.csv` is
  `MEAMA/synapse_pdist.csv` (the root folder had none; `hpc/synapse_pdist.csv`
  holds the same numbers with CRLF line ends). `aggregate_sweep.py` is
  unchanged: the cluster's copy was not in the zip (section 8). The
  pipeline files keep their non-ASCII comment characters (valid UTF-8, as on
  the cluster), so their sha256 still matches the cluster's.
- Registry (37-D). `PARAM_NAMES` gains `O_N` at index 36 (unit `1/(uM*s)`,
  nominal 0.3, in `ASTRO_PARAMS`); `PARAM_BOUNDS` row 36 is (0.03, 3.0),
  log-uniform; the `I_bias` box is (0.3, 1.0) [was (0.3, 1.5)];
  `assert PARAM_BOUNDS.shape == (37, 2)`. `HPC_main_sweep.py` repairs a stale
  36-D registry from `HPC_single_run.py` at import (appends `O_N`, nominal
  0.3). The model declares `O_N : 1/mole/second` per astrocyte, default
  `Params_astroGT['O_N']`, and run_args sets it to `params[36] / umole /
  second`.
- Sweep groups: `tripartite` = the free neuron, synapse and astrocyte axes
  minus `I_Theta` and `omega_I` (held at their nominals 0.3 uM and 0.05 uM):
  32 axes. `neuron_synapse` 23 (unchanged), `all` 34.
- Astrocyte contact gate (a topology-level axis). CLI of the sweep:
  `--stoa_gate_lo`, `--stoa_gate_hi` (default 1.0 each: off) and
  `--seed_stoa_gate` (default 71); of the single run: `--stoa_gate_p`,
  `--seed_stoa_gate`.
  - Per topology k: under mode Full with lo < hi, p_k ~ U[lo, hi] drawn from
    the task's master RNG (only then, so gate-off and Neuronal campaigns keep
    a byte-identical seed stream); with lo == hi, p_k = lo; under Neuronal
    the gate is not drawn, is declared inert and is held at 1.0. The prior
    must satisfy 0 <= lo <= hi <= 1 (else ValueError).
  - The gate draws come from a dedicated generator,
    `default_rng((topo_seed_base + seed_stoa_gate) mod (2^31 - 1))`, never
    from the position or wiring streams.
  - `build_topology(..., stoa_gate_p, seed_stoa_gate)`: every viable
    (synapse, astrocyte) pair of the topology rule is kept independently with
    probability p_k; both directions (spillover synapse -> astrocyte and
    gliotransmission astrocyte -> synapse) are built from the same kept list;
    p_k >= 1 draws nothing (output bit-identical to pre-gate builds); p_k
    outside [0, 1] raises ValueError. `StoA_viable_i/j` keep the pre-gate list.
  - Records: `stoa_gate_p` in each simulation's npz, sidecar and failure
    record; `seed_stoa_gate` and `n_stoa_viable` in `topology_meta.json`; the
    keys are absent under Neuronal. `manifest_version` 5.
- Unchanged by this block: the inverse-transform (BINV) `Binomial_fun` of
  the asynchronous release (python, cython and cpp implementations), Block
  1's hold, the CAdEx equations, the seed chain.
- Callers on `main`: `launch_campaign.sh` and `submit_sweep_mixed.sh` pass
  the gate flags since 0d8a1c9; with the 36-D root the worker rejected them
  as unrecognized arguments, now the two agree.
- `MEAMA/` launch code follows the cluster's copies: `submit_MEAMA.sh`,
  `bench_sizing.sh`, `env_check.py`, `make_brian_final.sh` byte-identical to
  them (the environment fixes of 2026-08-31: `set +u` around the conda
  calls, the interpreter from `CONDA_PREFIX`, the brian2/python profiles);
  `launch_MEAMA.sh` keeps Block 7's guard and takes the cluster's queues
  (intel:48:2, cpu:48:2; cfd commented out) and default walltime 300:00:00.
- Test oracles: `smoke_test_stoa_gate.py` (36 checks) and
  `smoke_test_param_recording.py` (119 checks, 11 astrocyte axes) on the root;
  Block 6's smoke test (R1 the 24-dimension fraction, R3 the 41 values of a
  `--rc-like` rebuild, R6 the tiny end-to-end run); Block 5's report on the
  repository (both CAdEx copies CLAMPED); an end-to-end Full `tripartite`
  sweep with the gate.
- Confirmed: yes 2026-10-31 that this version, with the contact gate and 37
  parameters, is the default (author); inferred for the content of this
  entry, read from the cluster's code, to be confirmed with the model's
  equations (section 8).
- Status: smoke-tested 2026-10-31 in the sandbox. `smoke_test_stoa_gate.py`
  36 passed (it failed on the 36-D root); `smoke_test_param_recording.py` 119
  passed; Block 6's smoke test 65 passed; Block 5 report: 2 CLAMPED, 1 SKIP;
  audit smoke tests 18 and 41 passed; seed allocation 47; burst metrics
  passed. End to end (`HPC_main_sweep.py`, Full, `tripartite`, 20 + 20
  cells, 2 topologies, 0.5 s, seed 4242): 32 free axes including `O_N`;
  gate p = 0.737 and 0.648 kept 184 of 254 and 59 of 92 viable links; 2 runs
  ok per topology; `manifest_version` 5; theta of length 37; `stoa_gate_p`
  in the npz and the sidecar. `APPLY_LCDUP.txt` steps 2-7 and Block 6's
  corners and interior ran on a LindaColloDUP folder built from the cluster
  files (Block 4). `smoke_test_cadex.py` fails stage 1 as before (section 8).
  `MEAMA/launch_MEAMA.sh` dry run on a mock folder: the guard's scan, intel
  and cpu at 48 cores, walltime 300:00:00. Skipped: davinci.

### Block 9: GiuliaGate launch package (Giulia's new campaign)

- Purpose (decided 2026-10-31, section 4): launch Giulia's new campaign on
  davinci from `hpc/Phenomenological_finalv1/GiuliaGate/` in the clone:
  MODE Full on the root pipeline (Block 8, with Block 1's hold), neurons at
  the default cell, the astrocyte contact gate drawn per network, 15 free
  axes.
- Modules: `prior_config.tsv` (the prior), `apply_prior_patch.py` (its only
  reader; standard library), `apply_freeze_patch.py` (Block 4's, a byte
  copy), `launch_GG.sh` (login node), `submit_GG.sh` (PBS worker),
  `seed_alloc.py` (Block 7's, a byte copy), `smoke_test_gg.py`,
  `APPLY_GG.txt`, `.gitignore` (the pipeline copies, `campaign_*/`,
  `artifacts/`, the patchers' backups). The pipeline files are copied in
  from the root folder (`APPLY_GG.txt` step 0) and patched in place.
- `prior_config.tsv`: ASCII, LF, tab-separated; `#` lines are comments; the
  header `axis group unit status low high value scale registry giulia_copy
  meaning`; one row per registry axis (37) plus `conn_prob` and
  `stoa_gate_p` (group `topology`). status `free` (low < high, natural
  units; value `-`) or `fixed` (a number in value; low, high `-`). scale is
  derived, not read: `log` iff low > 0 and high/low >= 10 (the rule of
  `HPC_main_sweep.py`), else `lin`. `apply_prior_patch.py` refuses (exit 1,
  nothing written) a TSV that does not list exactly those axes once each, a
  status other than free or fixed, a free row without 0 < high - low, a
  fixed row without a number, DeltaT, VT, gL, I_Theta or omega_I (outside
  `tripartite`, so the freeze list cannot move them) not fixed at the
  registry value, a topology box outside 0 <= low <= high <= 1, and a file
  with CR bytes.
- `apply_prior_patch.py` modes: none: every free axis whose `PARAM_BOUNDS`
  row differs from the TSV gets the TSV box, the row marked
  `[GiuliaGate box 2026-10-31] registry (lo, hi)`; timestamped backup,
  temporary file, re-read, compile, restore on failure; a second run prints
  `ALREADY APPLIED` and writes nothing; a file patched to other boxes is
  refused (copy it in again). `--check` (validate, print what would change);
  `--check-applied` (exit 0 iff every free row equals its TSV box; prints
  `BOXES OK: 15 free axes ...; changed from the registry: ...`);
  `--freeze-list` (every fixed axis of `tripartite`, `NAME=VALUE` as written
  in the TSV, comma-separated); `--topology NAME` (`LOW HIGH`);
  `--self-test` (12 checks). Options `--tsv`, `--file`, `--nominal`.
- Decided values (section 4). Free (15): Sigma [2, 10] mV (lin); U_0_sr
  [0.1, 1]; Omega_f_sr [0.1, 4.5] 1/s; Omega_d [0.1, 4.5] 1/s; alpha_syn
  [0.1, 1]; g_ampa, g_nmda [0.05, 10] nS; x0 [0.05, 0.5]; O_G [0.1, 10]
  1/(uM s); Omega_G [0.001, 0.1] 1/s; I_bias [0.3, 1] uM (lin); F [0.1, 10]
  uM/s; U_A [0.1, 0.9] (lin); G_T [50, 1000] mM; O_N [0.03, 3] 1/(uM s); log
  unless marked lin; the registry boxes except Sigma (registry [3, 15]) and
  g_ampa, g_nmda (registry [0.01, 5]). [corrected 2026-10-31] O_N [0.0001, 1]
  1/(uM s), log (registry [0.03, 3]; section 4). Fixed at the registry nominals:
  gbarA 0.9 nS, delta_gA 0.135 nS, tauA 200 ms, VA -48 mV, DeltaA 5 mV, VR
  -58.2 mV, I_inj 4 pA, Cm 17.25 pF, DeltaT 2 mV, VT -51 mV, gL 1.15 nS,
  EC50_ampa 8.6 mM, EC50_nmda 3 mM, U_0_ar 0.003, U_max 0.5 1/ms,
  Omega_f_ar 1.42857 1/s, O_beta 1 uM/s, O_3K 4.5 uM/s, Omega_5P 0.1 1/s,
  I_Theta 0.3 uM, omega_I 0.05 uM, C_Theta 0.5 uM. Topology: conn_prob
  [0.05, 0.4], stoa_gate_p [0.2, 1].
- Launcher (`launch_GG.sh`): reads the freeze list, the boxes changed from
  the registry and both topology boxes through `apply_prior_patch.py`
  (python3 on the login node); refuses (exit 2) a missing pipeline file,
  worker, patcher or TSV, a missing python3, a copy without `--freeze`, a
  copy failing `--check-applied`, a topology value that is not a number;
  (exit 3) SIZING_CONFIRMED != 1, REFRACTORY_CLAMP neither yes nor no;
  (exit 4) a model with other than one dV/dt line, REFRACTORY_CLAMP not
  matching it, a density <= 0, a topology box outside 0 <= LOW <= HIGH <= 1,
  a freeze list that is not `NAME[=NUMBER]` items, an existing
  `campaign_<tag>` folder (not in a dry run); (exit 5) no seed range from
  `seed_alloc.py` or an allocator without `--scan-tree`. Settings: MODE
  Full, SWEEP_GROUP tripartite, CONN_RULE flat, C_MAX 300 um, SIMTIME 180 s,
  DENSITY 1200 and DENSITY_ASTRO 1200 per mm^2 (Nn = Na = 108),
  N_TOPOLOGIES_WORKER 48 (an estimate), K_PARAMS 1, WALLTIME 24 h, TARGET
  10^6, HEADROOM_PCT 200, QUEUES intel:48:2 and cpu:48:2, CAMPAIGN_TAG
  `cadex_hhgap_clamp_v1`, CONDA_ENV `brian_final`; ntasks per queue =
  ceil(TARGET w_q / sum w x HEADROOM / (N_TOPOLOGIES_WORKER x ncpus x k)),
  w_q = ncpus x concurrency (435 per queue with the defaults). The seed
  guard as Block 7 (SEED_SCAN_TREE default the cluster's ANN). `qsub -v`
  carries REPO_DIR, NODETAG, SEED_BASE, CONDA_ENV, LAUNCH_ID, CAMPAIGN_TAG,
  N_TOPOLOGIES, N_PARAMS_PER_WORKER, C_MAX, SIMTIME, DENSITY, DENSITY_ASTRO,
  MODE, SWEEP_GROUP, FREEZE_ENC (`:` for `,`, `@` for `=`), CONN_RULE,
  CONN_PERIODIC, CONN_PROB_LO/HI, STOA_GATE_LO/HI. A launch writes
  `campaign_<tag>/PROVENANCE_<launch id>.txt` with the settings and the
  sha256 of the pipeline files, the TSV, the patcher, both scripts and the
  allocator. DRYRUN=1 prints the qsub lines and submits nothing.
- Worker (`submit_GG.sh`, `#PBS -N gg_campaign`, `-k eo`): refuses (exit 2)
  no NODETAG, no FREEZE_ENC, a REPO_DIR without `HPC_main_sweep.py` or
  `apply_prior_patch.py`, a copy without `--freeze`, a SEED_BASE that is not
  a non-negative integer, a failed `conda activate` or environment check,
  and a copy failing `--check-applied` (run with the environment's python,
  `$CONDA_PREFIX/bin/python`; with GG_NO_ENV=1 the ambient `python`, for the
  probe). Calls `HPC_main_sweep.py` with the injected settings, NN = NA =
  115 (overridden by the densities), displ_bias 20, topology_mode wallach,
  gj_dist 200, gj_max_dist 150, stoa_cutoff 70, stoa_sigma 200,
  seed_stoa_gate 71, seeds 50 / 39 / 35 / 60, dpi 150, `--seed_master
  SEED_BASE + PBS_ARRAY_INDEX`, then `--rebuild_manifest_only`.
- Test oracles (`smoke_test_gg.py`, G1-G8; 32 passed and 1 skipped without
  `--sim`, 35 with it): G1 the TSV (self-test, the 39 rows, 15 free axes,
  the scale rule); G2 the patches on a fresh copy (PATCHED with Sigma,
  g_ampa, g_nmda [and O_N since 2026-10-31]; ALREADY APPLIED; BOXES OK;
  compiles; 3 marked rows [4 since 2026-10-31]); G3
  the patched module (N_DIMS 37; tripartite minus the freeze list = the 15
  TSV axes; their boxes; LOG_PARAMS); G4 refusals (a TSV edited after
  patching; DeltaT off the registry); G5 the launcher's dry run (banner,
  two qsub lines with every injected value; refusals); G6 the worker with a
  stub python (`--check-applied` first; the arguments; refusals); G7 bytes;
  G8 the real worker on a tiny Full network (2 runs ok, the gate p in every
  sidecar, no neuron NaN warning; it prints the astrocyte NaN warnings).
  [corrected 2026-10-31] G8 also requires no astrocyte NaN warning (Block
  10); 35 passed with `--sim`.
- Confirmed: yes 2026-10-31 for the prior, the folder, MODE Full, 180 s,
  the densities and the gate (section 4); inferred for the sizing
  (N_TOPOLOGIES_WORKER 48 from MEAMA's measured 22.5 min per topology at
  ~3,300 synapses and 200 s, scaled to ~2,600 synapses and 180 s: ~16 min,
  so ~13 h per task; not measured), the tag, TARGET and HEADROOM (as
  MEAMA), the queues and the probe.
- Status: smoke-tested 2026-10-31 in the sandbox (35 passed with `--sim`).
  Skipped: davinci. Blocked for the campaign by section 8's astrocyte NaN:
  in G8 and in every Full run at nominal parameters the astrocytes go
  non-finite within about 30 ms. [corrected 2026-10-31] Unblocked by Block
  10 (G8: 0 astrocyte NaN warnings).

### Block 10: the astrocyte's glutamate drive is the mean over its links

- Purpose (decided by the author on 2026-10-31, section 4): the fix of the
  astrocyte NaN of section 8. Module: `Synapse_to_astro` in
  `hpc/Phenomenological_finalv1/ASD_fun_BD_cpp.py` (the root model, so every
  Full campaign copied from it: GiuliaGate, MEAMA's rerun, `launch_campaign.sh`).
  Not changed: the repository's `Giulia/ASD_fun_BD_cpp.py` copy, the
  Hodgkin-Huxley `hpc/ASD_fun_BD_cpp.py` and the top-level legacy
  `ASN_fun*.py`, which keep the sum (section 8).
- Equation. L is the list of (synapse j, astrocyte k) links of the topology
  (`build_topology`'s `StoA_i`, `StoA_j`, after the contact gate). For each
  astrocyte k with N_k = |{j : (j, k) in L}| >= 1 and for all t:
  Y_extra,k(t) = (1/N_k) sum over {j : (j, k) in L} of Y_A,j(t); for N_k = 0,
  Y_extra,k(t) = 0. Y_A,j is the slow cleft glutamate of synapse j (mM in
  the model's concentration convention). Brian2: `Y_extra_post = Y_A_pre /
  N_incoming : mole (summed)`, N_incoming = the number of links onto the same
  astrocyte. Until 2026-10-31 it was the sum (Y_extra_post = Y_A_pre).
  Reference: Stimberg et al. 2019, Computational Glioscience ch. 18, p. 498,
  `Y_S_post = Y_S_pre/N_incoming` (project knowledge).
- Stability (own derivation): dGamma_A/dt is linear in Gamma_A with rate
  lambda_A = O_N (Y_bias + s Y_extra)^n + Omega_N (1 + zeta C/(C + K_KC));
  the group's explicit RK4 is stable for lambda_A dt < 2.785 (the stability
  interval of classical RK4 on the negative real axis; textbook, from
  memory). Under the mean, Y_extra,k(t) <= max_j Y_A,j(t), and each release
  adds rho Y_T r_Sr <= rho Y_T = 30 mM to Y_A,j (r_Sr = u_sr x_S <= 1), so
  Y_extra stays near or below ~30-34 mM (measured maximum 29.9 mM at the
  high-drive corner of section 8). With s = 0.75, n = 1, Y_bias = 0 and
  dt = 0.05 ms that gives stability for O_N below about 2.2 per (uM s);
  GiuliaGate's O_N <= 1 has the bound at 74 mM. A box reaching O_N = 3 is not
  safe under the mean alone (section 8: NaN at 7 ms at that corner).
- The contact gate under the mean: p sets N_k and which synapses are
  linked (and, through the same list, which receive gliotransmitter); for
  synapses of similar activity it does not move Y_extra,k. Measured
  (sandbox, 108 + 108 cells, conn_prob 0.225, O_N 0.01, 5 s, astrocytes with
  at least one link, t > 1 s): mean Gamma_A 0.774, 0.774, 0.775 at p = 0.2,
  0.6, 1.0; linked astrocytes 83, 93, 94 of 108; median Y_extra 1.70 mM.
- Receptor operating point (same setting, gate 0.6): mean Gamma_A of the
  linked astrocytes 0.035, 0.262, 0.774, 0.910, 0.989, 0.997 at O_N =
  0.0001, 0.001, 0.01, 0.03, 0.3, 1 per (uM s); the steady state
  Gamma_A* = a/(a + b), a = O_N s Y_extra, b = Omega_N (1 + zeta C/(C +
  K_KC)), for Y_extra and C held fixed, gives the same numbers (algebra).
- Test oracles (`smoke_test_astro_drive.py`; 9 checks, 6 and 1 skipped
  with `--no-sim`): A1 the model line (the mean once, the sum
  gone); A2 the summed-variable semantics through the library's own
  `Synapse_to_astro` on a stand-in network (Brian2 runtime, numpy): means of
  known values to float64 roundoff, 0 for an astrocyte without a link, one
  link and four links of equal value give the same drive, the returned
  (astrocyte, synapse) pairs; A3 end to end, `HPC_single_run.py`, Full,
  nominal, 48 + 48 cells on 200 um, 3 s: exit 0, no NaN warning, events from
  at least half of the astrocytes (48 of 48). Planted fault (the sum
  restored, 2026-10-31): A1, both A2 value checks and both A3 checks fail (4
  NaN warnings, 1 astrocyte event). Regression on the root: Block 8's
  `smoke_test_stoa_gate.py` 36 and `smoke_test_param_recording.py` 119
  passed; Block 6's smoke test 87; Block 9's 35 with `--sim`.
- Custom code: none (a Brian2 summed variable).
- Confirmed: yes 2026-10-31 (author: "use the option a").
- Status: smoke-tested 2026-10-31 in the sandbox. Skipped: davinci.

### 5.1 Projects and where their model runs

The tester can check only what is in the repository. The cluster folders
below are separate copies (not git checkouts, `HPC_Repository_Hierarchy.md`
in project knowledge); Block 5 clamps them and its report, pasted by the
author, is the record for them.

| project | campaign family (job name) | model file it runs | in the repository | how it gets the clamp |
|---|---|---|---|---|
| Giulia's simulations | `campaign_cadex_hhgap_*` (`asn_campaign`) | `Phenomenological/Main/Giulia_Astro/ASD_fun_BD_cpp.py` (worker `lib_dir` = launch folder) (inferred) [corrected 2026-10-31: the new campaign, `cadex_hhgap_clamp_v1` (`gg_campaign`), runs the root pipeline copied into `Astro-Neuron-Network/hpc/Phenomenological_finalv1/GiuliaGate/` in the clone, Block 9] | `hpc/Phenomenological_finalv1/Giulia/` (relationship to `Giulia_Astro/` unresolved); the new campaign's launch package `GiuliaGate/` (Block 9) | repository copy: Block 1; cluster copy: Block 5; GiuliaGate copies the clamped root file (step 0), and its launcher refuses a mismatch with `REFRACTORY_CLAMP` |
| rho1300 family, the 100-electrode project (confirmed 2026-10-30; 637 neurons, `C_MAX` 700 um in `submit_sweep_mixed.sh`) | `campaign_cadex_rho1300v*` (`asn_campaign`) | `Phenomenological/Main/ASD_fun_BD_cpp.py` (inferred) | root pipeline `hpc/Phenomenological_finalv1/` (the cluster copy may drift) | repository copy: Block 1; cluster copy: Block 5 |
| MEAMA | `campaign_meama_rho1600_*` (`meama_campaign`) | `Phenomenological/MEAMA/ASD_fun_BD_cpp.py`, 37-D | launch code only (`hpc/Phenomenological_finalv1/MEAMA/`); the 37-D model is not in the repository [corrected 2026-10-31: the 37-D model is the root pipeline, Block 8; the cluster's copy is the same file without the hold] | Block 5 |
| LindaColloDUP | `campaign_lcdup15_*` (`lcdup_campaign`) | `Phenomenological/LindaColloDUP/ASD_fun_BD_cpp.py`, copied from `MEAMA/` [corrected 2026-10-31: copied from the clone's root folder, Block 8, already clamped] [corrected 2026-10-30: `Astro-Neuron-Network/hpc/Phenomenological_finalv1/LindaColloDUP/ASD_fun_BD_cpp.py` in the git clone, copied from `MEAMA/` and ignored by git; Block 5 sees it as an ordinary file, not GIT CLONE, since it is untracked] | launch package (Block 4); the model file is not in the repository | inherits it from a clamped `MEAMA/` copy, or Block 4's `apply_refractory_clamp.py` (step 5a); its launcher refuses a mismatch with `REFRACTORY_CLAMP`. On a copy clamped by Block 5, step 5a prints `ALREADY CLAMPED` with exit 0 for `--check` and the patch, and the launcher reads the file as clamped (sandbox, 2026-10-30) |

After each relaunch, Block 3's CODE section, run on the new campaign
folders, must report CLAMPED for each (it reads the model file named by the
campaign's `job_args.json` `lib_dir`).

## 6. Pipeline entry points

| Command | Purpose | Minimal configuration for testing | Expected runtime |
|---|---|---|---|
| `python smoke_test_refractory_scan.py` | Blocks 2 and 3 | none (synthetic campaigns) | seconds |
| `bash check_refractory_nan.sh --sample 300` | audit of existing campaigns | a campaign folder with `job_args.json` and `topo_*/iter_*.npz` | minutes (not timed on davinci) |
| `qsub -q cpu submit_refractory_scan.sh` | per-run tables for every run | -- | up to 12 h |
| `python apply_refractory_clamp.py --self-test` | Block 4 clamp patcher | embedded fixture | seconds |
| `python apply_freeze_patch.py --self-test` | Block 4 freeze patcher | embedded fixture | seconds |
| `python apply_box_patch.py --self-test` | Block 4 g box patcher | embedded fixture | seconds |
| `python smoke_test_freeze.py` | Block 4, both patches and the axis set | the LindaColloDUP folder after both patches (numpy, matplotlib) | seconds |
| `SIZING_CONFIRMED=1 REFRACTORY_CLAMP=yes DRYRUN=1 bash launch_LCDUP.sh` | Block 4 dry run | the LindaColloDUP folder on davinci | seconds |
| `python3 clamp_cadex_copies.py --self-test` and `python3 smoke_test_clamp_cadex_copies.py` | Block 5 | embedded fixtures; a repository clone for the mock tree | seconds |
| `python3 clamp_cadex_copies.py <repository root>` | the repository invariant of Block 1 | the repository | seconds |
| `python3 smoke_test_seed_allocation.py` | Block 7 (and the allocator's older groups) | numpy, the root pipeline | seconds |
| `python3 smoke_test_stoa_gate.py` | Block 8: the contact gate (36 checks) | the root folder | seconds |
| `python3 smoke_test_param_recording.py` | Block 8: axis declaration and records under the 37-D registry (119 checks) | the root folder | seconds |
| `python3 seed_alloc.py --root . --ledger artifacts/seed_ledger.tsv --scan-tree /davinci-1/home/ldellamea/ANN --report` | Block 7 on davinci: what the guard sees | a launch folder | seconds (not timed on davinci) |
| `python3 smoke_test_range_check.py` | Block 6 | the repository pipeline, Brian2 and a C++ compiler | minutes |
| `qsub -v SETUP=...,LIB_DIR=...,OUT=... submit_range_check.sh` | Block 6 on a setup, one node | the setup's folder on davinci | hours (not timed on davinci) |
| `SETUP=lcdup bash launch_range_check.sh` (first with `DRYRUN=1`) | Block 6 on a setup: corners and interior on two nodes, then the analysis job | the setup's folder on davinci; `qsub` | seconds to submit; the jobs: 352 (corners) and 343 (interior) runs per worker, about 11 h per node at the sandbox core's speed (not timed on davinci; `claude/Prior_Range_Calibration_Standard.md` section 6) |
| `python3 range_check.py --rc-phase corners --rc-level 0 --rc-out X --rc-design-only <settings>` | Block 6: a level's design, nothing run | the setup's folder | seconds |
| `NODES=4 SETUP=giulia bash launch_range_check.sh` (first with `DRYRUN=1`) | Block 6 on GiuliaGate: corners split by conn_prob level on two intel nodes, interior split in two on two cpu nodes, then the analysis job | the GiuliaGate folder after `APPLY_GG.txt` steps 0-4; `qsub` | seconds to submit; the jobs about 1-2, 9-10, 4 and 6-7 h (estimate from MEAMA's davinci throughput, not measured) |
| `python3 smoke_test_gg.py` (`--sim` adds G8) | Block 9 | the repository; G8 needs Brian2 and a C++ compiler | 30 s; about 1 min with `--sim` |
| `python3 smoke_test_astro_drive.py` (`--no-sim` skips A3) | Block 10 | the root folder; A3 needs Brian2 and a C++ compiler | seconds; about 35 s with A3 |
| `SIZING_CONFIRMED=1 REFRACTORY_CLAMP=yes DRYRUN=1 bash launch_GG.sh` | Block 9 dry run | the GiuliaGate folder after steps 0-4 | seconds |

## 7. Not executable in the sandbox

- Anything on davinci: PBS submission, the 37-D pipeline files (`MEAMA/`,
  `LindaColloDUP/`), real campaign sizes, job-log naming, read speed.
  Substitutes used: 36-D `main` copies with no-op parser options for the
  37-D flags; a stub `qsub`; campaign trees built from real worker output.
  [corrected 2026-10-31] The root is the 37-D pipeline (Block 8): no
  substitute is needed for it, and LindaColloDUP's procedure was re-run on
  the cluster's own files.
- Runs at the campaigns' length (180-200 s); the checks used 5 s, and 30 s
  for false positives.

## 8. Open questions

- Confirm Block 2's thresholds (t_early = 1 s; the late rule) (raised 2026-10-30).
- LindaColloDUP: the VA/VR option and I_inj (raised 2026-10-29).
  [closed 2026-10-30] VA -48 mV, VR -58.2 mV, I_inj 4.0 pA (section 4).
- MEAMA: a clamped campaign in a folder of its own, or not (raised 2026-10-30).
  [closed 2026-10-30] Superseded: the reruns are launched from folders of the
  git clone (section 4); MEAMA's folder there is chosen when it is prepared.
- LindaColloDUP at the low end of its conn_prob box (noted 2026-10-30, for
  the range check to answer). Under the flat rule each directed pair is
  wired independently, so a neuron's in-degree is Binomial(Nn - 1,
  conn_prob). At conn_prob 0.01 and Nn = 159: mean in-degree 1.58,
  20.4 % of the neurons (32.5 per network) receive no synapse, and the
  largest strongly connected group holds 37 % of the neurons (weakly
  connected: 95 %); at 0.02: 4.1 % without input, 91 % strongly connected;
  at 0.05: 0.05 neurons per network without input, 99.9 %. 8.2 % of the
  topologies drawn from U[0.01, 0.5] lie below 0.05. (Exact binomial values
  and a Monte Carlo of the flat rule, 2000 networks per value, sandbox.)
  Those networks may be driven mostly by the per-neuron bias and the noise
  rather than by recurrence; the oat level 0.01 shows it.
- The pipeline files of the LindaColloDUP folder in the clone are copies
  outside git until the author sends them; versioning them in a separate
  repository folder, matched to a campaign through the sha256 values in its
  PROVENANCE file, would bring them under this spec (raised 2026-10-30).
  [2026-10-31] They are now copies of the tracked root files (step 0),
  changed only by the tracked patchers; the PROVENANCE file's sha256 values
  tie a campaign to them.
- Block 6's `giulia` setup reads `LIB_DIR` = `Phenomenological/Main/Giulia_Astro`
  and its settings from that folder's `campaign_cadex_hhgap_v5`, both outside
  the clone; it is replaced by the new Giulia folder's own settings when that
  folder is built (section 4, raised 2026-10-31). [closed 2026-10-31] The
  giulia setup is GiuliaGate and reads its prior from `prior_config.tsv`
  (Block 6).
- Where this project's decision log lives (raised 2026-10-29).
- The cluster folder of each project in section 5.1 (inferred paths) and that
  Block 5's report lists them all as CLAMPED (raised 2026-10-30).
- The 37-D model files of `MEAMA/` (and so of LindaColloDUP) are not in the
  repository, so the tester cannot check them; committing them would bring
  them under this spec (raised 2026-10-29, still open). [closed 2026-10-31]
  They are the root pipeline now (Block 8).
- Astrocyte variables ending NaN in Full runs at nominal parameters: a
  separate failure, not investigated (raised 2026-10-29). [diagnosed
  2026-10-31; sandbox, root pipeline at fbc20f6 via `HPC_single_run.py`,
  108 + 108 cells, conn_prob 0.225, gate 0.6, seed 7, every astrocyte
  recorded at every 0.05 ms step] Mechanism: an astrocyte's `Y_extra` is the
  sum of `Y_A` over its contacts (`Y_extra_post = Y_A_pre : mole (summed)`
  in `Synapse_to_astro`); each presynaptic spike adds rho Y_T r_Sr to `Y_A`
  (rho Y_T = 0.005 x 6000 mM = 30 mM), which clears with tau 30 ms, so
  during the onset burst `Y_extra` reached 0.1-1.3 M. The receptor equation
  dGamma_A/dt = O_N (Y_bias + spill_over Y_extra)^n (1 - Gamma_A) - Omega_N
  (1 + zeta C/(C + K_KC)) Gamma_A is linear in Gamma_A with decay rate
  lambda_A = O_N (Y_bias + spill_over Y_extra)^n + Omega_N (1 + zeta C/(C +
  K_KC)); the group's explicit RK4 is stable for lambda_A dt < 2.785, so,
  with n = 1, dt = 0.05 ms and Y_bias = 0, only while spill_over Y_extra <
  2.785 / (dt O_N): 186 mM at the nominal O_N = 0.3 /(uM s), 18.6 mM at
  O_N's upper bound 3. Past it Gamma_A oscillated with growing amplitude
  (x15-20 per step) to -inf (float32, the group's dtype) at 26.7 ms in the
  first astrocyte (`Y_extra` had reached 483 mM), then NaN, and the NaN
  spread through `I_coupling_tot` (gap junctions) to all 108 astrocytes by
  30.6 ms. Effect: for the rest
  of the run no astrocyte event (C > C_osc is false for NaN) and so no
  gliotransmitter release; the neurons stay finite (no NaN path leads back
  to them) and run without astrocyte feedback. At nominal parameters every
  Full run is affected, MEAMA's campaigns included (same files, Block 8;
  not checked on their data). The project's reference implementation drives
  the astrocyte with the mean over its synapses, `Y_S_post =
  Y_S_pre/N_incoming : mmolar (summed)` (Stimberg et al. 2019, Computational
  Glioscience ch. 18, p. 498; project knowledge), where this model sums.
  Candidate fix, tested in scratch copies (not committed): (a) the mean,
  `Y_extra_post = Y_A_pre/N_incoming : mole (summed)`; (b) Gamma_A advanced
  once per step by its exact solution for Y_extra and C held over the step
  (the Rush-Larsen update), Gamma_A <- Gamma_inf + (Gamma_A - Gamma_inf)
  exp(-lambda_A dt) with Gamma_inf = O_N (Y_bias + spill_over Y_extra)^n /
  lambda_A, as a `run_regularly` before the state update, Gamma_A removed
  from the RK4 system. Results (5 s at nominal; 3 s at a high-drive corner:
  O_N 3, U_0_sr 1, g_ampa = g_nmda = 10 nS, Omega_d 4.5 /s, x0 0.5,
  conn_prob 0.4, gate 1): unchanged model, NaN from 27 ms and 0 astrocyte
  events; (a) alone finite at nominal but NaN from 7 ms at the corner
  (Y_extra up to 29.9 mM, beyond the 24.8 mM bound at O_N = 3); (b) alone
  finite in both, Y_extra up to 6 M; (a) + (b) finite in both, Y_extra up
  to 30 mM, Gamma_A in [0, 1], one calcium event per astrocyte in 5 s (C
  peaking at 0.6-0.9 uM about 2 s in). Decision pending with the author;
  until then Full-mode astrocyte outcomes carry no information. Also for
  the author: with either drive Gamma_A stays near saturation (mean 0.85-0.9)
  once the network is active, since O_N spill_over Y_extra >> Omega_N for
  Y_extra in the mM range, so O_N moves the astrocyte little; a calibration
  question for the receptor and for `spill_over`. [corrected 2026-10-31] The
  0.85-0.9 averaged in the astrocytes without any link (Gamma_A = 0 for all
  t; 15 of 108 here): over the linked astrocytes, t > 1 s, Gamma_A is 0.989
  at O_N 0.3 and 0.999 at O_N 3 (mean drive). [closed 2026-10-31] Fixed by
  option (a), Block 10, decided by the author; option (b) not adopted. Under
  the mean drive alone O_N must stay below about 2.2 per (uM s) (Block 10);
  GiuliaGate's O_N box is [0.0001, 1], which spans the receptor's curve at
  nominal activity (Gamma_A 0.035 to 0.997, Block 10). The saturation
  question remains for boxes that start at O_N >= 0.03 (MEAMA's registry
  box [0.03, 3], which also crosses the stability bound).
- Copies of the model outside Block 10 keep the summed drive: the
  repository's `Giulia/ASD_fun_BD_cpp.py` (37-D, synapse_astro), the
  Hodgkin-Huxley `hpc/ASD_fun_BD_cpp.py` and the top-level `ASN_fun*.py`
  (Y_S summed). If the cluster's `Giulia_Astro/` copy sums too, Giulia's
  earlier campaigns (`campaign_cadex_hhgap_v1`-`v5`) may have lost their
  astrocytes the same way; not checked on their data, and that copy is not
  in the repository (raised 2026-10-31).
- The legacy Izhikevich model in the top-level `ASN_fun_pheno.py` has a 2 ms
  refractory period and no hold: during it the quadratic term can carry v
  past the threshold with no reset, as in the CAdEx runaway. Not used by any
  campaign; fix only if that code is revived (hold v at c, or drop the
  refractory period, as Izhikevich 2003 has none) (raised 2026-10-30).
- `launch_campaign.sh` and `launch_MEAMA.sh` still fall back to raw OS
  entropy, with a warning, when no seed allocator is found; the
  LindaColloDUP launcher refuses instead. Make them refuse too when their
  reruns are prepared (raised 2026-10-30).
- Retune the prior boxes of MEAMA and of the rho1300 family for the clamped
  model, after Block 6 has run on them (decided 2026-10-30, not started).
- Bring the Giulia setup onto GitHub: its 37-D pipeline with the new axis,
  the probability that a synapse is contacted by an astrocyte (the contact
  gate), which today exists only in the cluster's `Giulia_Astro/` (decided
  2026-10-30; files to come from the cluster). [2026-10-31] MEAMA's gate
  version is the default now (Block 8); whether `Giulia_Astro/` runs the
  same code is known only once its files arrive. [2026-10-31] Superseded by
  the decision to run everything from the clone (section 4): Giulia's new
  campaign runs the root pipeline. Its configuration is proposed in
  `hpc/Phenomenological_finalv1/GiuliaGate/` (`prior_config.tsv`,
  `campaign_config.tsv`, rendered in `CONFIG_SHEET.md`; read by no code
  yet): Giulia's design (Full; neurons fixed; the 22 axes of her group
  `synapse_astro` drawn) plus the contact gate and the hold. Open for the
  author: the cell (Giulia's Gorski adaptive-spiking cell, which needs a
  patch for El, t_ref, gl_ref and the frozen VT and gL rows, or the
  registry cell), Sigma, I_inj, the boxes, the gate box, the culture size,
  the folder and the tag. The launch package and its block follow the
  decisions. [closed 2026-10-31] Decided (section 4): the registry cell,
  Sigma free in [2, 10] mV, I_inj fixed at 4 pA, the boxes, the gate box
  [0.2, 1], 300 um at 1200 + 1200 per mm^2, the GiuliaGate folder, the tag
  `cadex_hhgap_clamp_v1`; the launch package is Block 9
  (`campaign_config.tsv` and `CONFIG_SHEET.md` removed, `prior_config.tsv`
  read by `apply_prior_patch.py`).
- Topology-level axes other than conn_prob (the contact gate of the new
  Giulia setup) are not design dimensions of Block 6 yet: with a gate box in
  EXTRA each topology draws its gate from the prior (raised 2026-10-30).
  [2026-10-31] The giulia setup reads the gate box from `prior_config.tsv`;
  the gate stays drawn per network as in the campaign (decided), and its
  effect is read across networks (`gate_networks.csv`).
- `HPC_single_run.py`'s standalone command line maps its 30 flags onto the
  first 30 slots of the 37-D vector by position: slots 1, 4, 12, 13 and 16
  carry the old names (`--g_AHP`, `--Tau_Ca`, `--g_na`, `--g_kd`,
  `--alpha_Ca`) of what are now gbarA, tauA, DeltaT, VT and delta_gA, and
  slots 30-36 (gL, VA, DeltaA, VR, I_inj, Cm, O_N) have no flag. The sweep
  does not use it (it builds run_args from the vector); only standalone
  single runs are affected (raised 2026-10-31).
- Block 6 for the 37-D setups: the driver uses only `main`, `sample_theta`,
  `resolve_sweep_group`, `_apply_freeze`, `_resolve_n_workers`,
  `NOMINAL_PARAMS`, `PARAM_BOUNDS_THETA`, `natural_to_theta`,
  `theta_to_natural`, `build_parser` (and its `conn_prob_lo`, `conn_prob_hi`,
  `conn_rule`), `PARAM_NAMES`, `LOG_PARAMS` and `N_DIMS` of the setup's module;
  checked on the 36-D root pipeline and the 37-D `Giulia/` copy, not on the
  cluster's gate version (raised 2026-10-30). [closed 2026-10-31] Checked on
  the gate version: Block 6's smoke test on the 37-D root, and a corners and
  an interior level on a LindaColloDUP folder built from MEAMA's files.
- `aggregate_sweep.py --write-index` puts `conn_prob` but not `stoa_gate_p`
  in its index table, so a gated campaign's index lacks the gate value (it is
  in every sidecar and npz). The cluster's `MEAMA/aggregate_sweep.py` was not
  in the zip of 2026-10-31 and may differ (raised 2026-10-31).
- `smoke_test_cadex.py` (root) checks a 35-D registry: its stage 1 failed on
  `main` before Block 8 (11 failures, 36-D) and fails after it (10, 37-D);
  its stage 2, a nominal Neuronal run, passes now that `synapse_pdist.csv`
  is in the root folder. Rewrite it for the 37-D registry or retire it
  (raised 2026-10-31).
- The model's equations (neuron, synapse with the BINV asynchronous release,
  receptors and cleft glutamate, astrocyte calcium and IP3,
  gliotransmission, gap junctions, the contact gate) and the rest of the
  sweep loop get spec entries, confirmed with the author, before the
  independent tester runs on Block 8 (decided 2026-10-31).
