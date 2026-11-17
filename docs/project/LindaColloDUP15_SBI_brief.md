# LindaColloDUP15 simulations (`lcdup15_rho1300_neur_n01`) -- specifics for SBI

| Date | Change |
|---|---|
| 2026-11-16 | Created for the SBI work. Built from the repository at `main` 1789e40 and at f216b00 (the code this campaign copied, section 1), the project-knowledge snapshot `claude/LindaColloDUP_Campaign_Operations.md` (last row 2026-11-04) and the davinci `qstat` output pasted by the user on 2026-11-05. Nothing was run on davinci or in the sandbox for this document. |
| 2026-11-17 | Corrections, prompted by the SBI request of 2026-11-17 and a re-read of f216b00 (`HPC_main_sweep.main`, `submit_LCDUP.sh`): 192 (cfd) or 48 (cpu) simulations share each topology, with different theta (sections 1, 2, 5); a run is also discarded when it raises (section 4); the SBI pairing with DUP15HD (section 7). Answers to that request, with reference sha256 values for the PROVENANCE file: `HANDOFF_2026-11-17_reply_to_SBI_LindDUP.md`. |

Basis tags: `[REPO]` read in the repository; `[KB]` the project-knowledge snapshot of the campaign's operations doc; `[CLUSTER]` davinci output pasted by the user; `[OWN]` own inference; `[EST]` own arithmetic or estimate.

Procedure and history: `hpc/Phenomenological_finalv1/LindaColloDUP/APPLY_LCDUP.txt`, `specs/SPEC.md` Block 4, and the operations doc named above.

## 1. Identity and status

| item | value |
|---|---|
| name | LindaColloDUP15 = campaign tag `lcdup15_rho1300_neur_n01` `[KB]` |
| output folder (davinci) | `~/ANN/Astro-Neuron-Network/hpc/Phenomenological_finalv1/LindaColloDUP/campaign_lcdup15_rho1300_neur_n01/` `[REPO]` |
| launched | 2026-11-02, between 09:02 and 09:23 UTC `[OWN]`, likely. On 2026-11-05 the arrays 1942752[] (queue cfd, 192 cores) and 1942753[] (queue cpu, 48 cores) were running `[CLUSTER]`. The cfd queue entered the launcher with `main` f216b00 (09:02 UTC), and both job IDs are below 1942983, the analysis job of the GiuliaGate range check submitted at 09:23 UTC. The campaign's `PROVENANCE_<launch id>.txt` settles it (section 7). |
| code that ran | the pipeline copied before that launch, so a commit between f216b00 and 004ed7b `[OWN]`, likely: 37-D parameter vectors, refractory clamp on, per-run seeds not applied (section 5). The PROVENANCE file holds its sha256 values. |
| jobs | queues `cfd:192:1` (163 tasks, `-J 0-162%1`) and `cpu:48:4` (652 tasks, `-J 0-651%4`); 32 topologies per worker, one parameter vector per topology. **[corrected 2026-11-17]** 32 topologies per task, not per worker; on each topology every worker runs one parameter vector, so 192 (cfd) or 48 (cpu) simulations share one topology and its conn_prob `[REPO, HPC_main_sweep.py lines 1643-1866, submit_LCDUP.sh lines 128 and 169-171]`. That makes 6,144 simulations per cfd task and 1,536 per cpu task, a capacity of 2.0e6 for a target of 1e6 (headroom 200 %) `[REPO]`. Logs `~/lcdup_campaign.o*`, `.e*`. |
| throughput | about 16,000-21,000 simulations a day, so 50-60 days to 1e6 plus queue waits `[EST]` (operations doc section 5; not measured) |
| state | running on 2026-11-05 `[CLUSTER]`; not checked since (section 7) |

## 2. One simulation

- **Mode.** Neurons only (`MODE=Neuronal`, `ASTRO_FRAC=0`). With no astrocytes, the astrocyte axes and alpha_syn, O_G and Omega_G are inert `[REPO]`.
- **Neurons.** 159 excitatory CAdEx neurons (conductance-based adaptive exponential integrate-and-fire, Gorski et al. 2021): 1,300 neurons/mm^2 on a 350 um square culture (`RHO_TOTAL 1300`, `C_MAX 350`) `[REPO]`.
- **Refractory clamp.** V is held at VR for t_ref = 3 ms (`(unless refractory)`) `[REPO]`.
- **Wiring.** Flat rule: every ordered pair (i != j) is connected independently with probability conn_prob. No distance dependence, not periodic; conn_prob is drawn once per topology `[REPO]`.
- **Synapses.** Tsodyks-Markram release with a synchronous and an asynchronous (binomial) branch. AMPA and NMDA receptors are driven by cleft glutamate through a Hill function; NMDA has the Jahr-Stevens Mg2+ block `[REPO]`.
- **Noise.** White noise on V, `noise = Sigma*(2*gl_ref/Cm)**0.5*randn()/sqrt(dt)`, so Sigma sets the voltage-noise amplitude in mV `[REPO]`.
- **Heterogeneity.** A frozen bias current `I_inj*bias_unit` per neuron, with bias_unit ~ U(-0.5, 0.5); V(0) ~ U[El, VT) `[REPO]`.
- **Length.** 180 s per simulation at dt = 0.05 ms. Every simulation has its own topology `[REPO]`. **[corrected 2026-11-17]** It does not: 192 (cfd) or 48 (cpu) simulations, each with its own theta, run on one topology, with the same conn_prob, wiring and positions `[REPO]`. Split held-out sets by topology folder `[OWN]`.

## 3. Parameters and prior

theta is 13 free axes of the `neuron_synapse` group plus conn_prob. Each free axis is drawn independently and uniformly in its inference coordinate: theta_j = ln(value) on log axes (natural log), the value itself on linear axes. So log axes are log-uniform in natural units `[REPO]`. The index is the position in this campaign's 37-D `params` and `theta` vectors.

| index | axis | unit | box | scale |
|---|---|---|---|---|
| 0 | Sigma | mV | [3, 15] | lin |
| 1 | gbarA | nS | [0.01, 10] | log |
| 4 | tauA | ms | [10, 2000] | log |
| 5 | U_0_ar | - | [1e-4, 0.05] | log |
| 6 | U_max | 1/ms | [0.1, 1] | log |
| 7 | U_0_sr | - | [0.1, 1] as launched; the registry box is [0.01, 0.7] since 2026-11-03 | log |
| 8 | Omega_f_sr | 1/s | [0.1, 4.5] | log |
| 9 | Omega_f_ar | 1/s | [0.1, 4.5] | log |
| 10 | Omega_d | 1/s | [0.1, 4.5] | log |
| 14 | g_ampa | nS | [0.01, 10] (folder's box patch; registry [0.01, 5]) | log |
| 15 | g_nmda | nS | [0.01, 10] (folder's box patch; registry [0.01, 5]) | log |
| 16 | delta_gA | nS | [0.005, 10] | log |
| 17 | x0 | - | [0.05, 0.5] | log |
| - | conn_prob | - | U[0.01, 0.4], per topology | lin; stored as `conn_prob`, not in `theta` |

**Fixed** `[REPO]`:

| quantity | value |
|---|---|
| Cm | 17.25 pF |
| I_inj | 4.0 pA |
| VA | -48 mV |
| DeltaA | 5 mV |
| VR | -58.2 mV |
| EC50_ampa | 8.6 (model glutamate units, mM) |
| EC50_nmda | 3.0 (model glutamate units, mM) |
| DeltaT | 2 mV |
| VT | -51 mV |
| gL | 1.15 nS |
| inert | alpha_syn 1, O_G 1.5, Omega_G 0.00833, and every astrocyte axis |

The model's other constants are in `ASD_fun_BD_cpp.py`. Do not pool these simulations with GiuliaGate's: GiuliaGate runs neurons with astrocytes (108 + 108 cells), with 38-D vectors and other boxes `[REPO]`.

## 4. Data on disk and the training index

Each simulation writes `sweep_<queue>_task<NNNN>/topo_<NNNNN>/iter_<NNNNN>.npz` (fields as of f216b00) `[REPO]`:

| key | content |
|---|---|
| `params` | 37 values, natural units |
| `theta` | 37 values, inference coordinates; the frozen axes are constant columns |
| `conn_prob`, `conn_rule` ('flat'), `conn_periodic` | topology |
| `spk_N_t`, `spk_N_i` | spike times (s, float32) and neuron indices |
| `topo_idx`, `seed_run`, `noise_mode` | provenance; `seed_run` was not applied (section 5) |
| `spk_A_t`, `spk_A_i` | empty (no astrocytes) |

Next to each npz:
- `iter_<NNNNN>.json` repeats the labels and adds scalars over the whole 180 s: `mean_FR_Hz`, `frac_active`, `across_cell_rate_cv`, `mean_isi_cv`, `t_run_s` `[REPO]`.
- Each topology folder has `topology.npz` and `topology_meta.json`.
- Each task folder has `job_args.json` (`_active_param_names`, `_axis_declaration`, the freeze list) and `manifest.json`.
- Failed runs are listed in `topo_*/_failures.jsonl`; a run is discarded only when its spike times are non-finite `[KB]`. **[corrected 2026-11-17]** Also when the run raises, e.g. a binary crash or a Brian2 error (`_worker_entry`, f216b00 lines 1113-1212) `[REPO]`.

The training index `[REPO]` (the burst metrics read every npz: run them in a job, not on the login node `[OWN]`):

```bash
cd ~/ANN/Astro-Neuron-Network/hpc/Phenomenological_finalv1/LindaColloDUP && conda activate brian_final
python aggregate_sweep.py campaign_lcdup15_rho1300_neur_n01 --write-index
python aggregate_sweep.py campaign_lcdup15_rho1300_neur_n01 --burst-metrics
```

`--write-index` writes `campaign_index/theta_all.npy` (N x 37), `npz_paths.txt` and `summary.csv` (with `conn_prob` and the scalars above). `--burst-metrics` adds the burst panel of `burst_metrics.py`. Train on the 13 columns named by `_active_param_names`, plus `conn_prob` if it is to be inferred. The script's docstring still describes an older campaign ("N, 35", "22 free axes").

## 5. Randomness and the seed defect

- **Fresh per simulation.** theta, conn_prob and the topology, drawn with NumPy from the launch's seed span `[REPO]`. **[corrected 2026-11-17]** Only theta is fresh per simulation; conn_prob and the topology are drawn once per topology and shared by its 192 or 48 simulations `[REPO]`.
- **Not fresh.** This campaign was launched before SPEC Block 11 (2026-11-04) `[OWN]`, likely (section 1). In that code the per-run seed never reached the compiled binary (`APPLY_LCDUP.txt` and SPEC Block 11 `[REPO]`; operations doc section 7 `[KB]`):
  - every simulation's in-simulation random draws (membrane noise, asynchronous release) start from the same seed, SEED_DEVICE 50;
  - V(0) and bias_unit are the same in every network.
- **Labels.** So `seed_run` is not the seed that ran, and `noise_mode` 'fresh' is not true. A simulation cannot be re-run from its recorded seed. It can be re-run with the same code, theta and topology, since the effective seed is the same for all `[OWN]`.
- **Not checked.** How alike the noise of two simulations is. Every simulation starts the same stream, but each has its own topology and the number of draws per step depends on the network. The realizations therefore probably fall out of step after the first steps `[OWN]`; this is a guess. The bias pattern is shared by all networks `[REPO]`. **[corrected 2026-11-17]** The 192 or 48 simulations of one topology share the topology, the build seed and so V(0) and the bias pattern, and differ only in theta. Their streams separate only once their states consume random numbers differently: `Binomial_fun` draws nothing when floor(x_S/x0) = 0 or u_ar dt = 0 (SPEC section 8, 2026-11-03) `[REPO]`. How soon that happens is still not checked.
- **Check on any file.** An npz written by this campaign has no `seed_verified` key, which is written only since Block 11 `[REPO]`.
- **Open, the user's decision.** Train on n01 as it is, or launch `lcdup15_rho1300_neur_n02` with the current pipeline (`APPLY_LCDUP.txt` steps 0-8). An n02 would differ in three ways: seeds applied, U_0_sr box [0.01, 0.7], and 38-D vectors (g_sic, inert here). Never pool n01 and n02 without aligning both `[OWN]`.

## 6. What the range check says about identifiability

These are the results of the standard range check of this setup `[KB]` `[CLUSTER]`: run 2026-10-31, `~/ANN/range_check/lcdup_20261031T053658Z/`, 20 s runs, window 2-20 s, with conn_prob up to 0.5 and U_0_sr [0.1, 1] at the time.

- **Strong axes.** Main effects on y = log10(rate + 0.01 Hz), upper minus lower bound: Sigma +1.73, gbarA -1.24, delta_gA -0.90, Omega_d +0.51, tauA -0.48, g_nmda +0.46, conn_prob +0.45, g_ampa +0.32. Largest interactions: Sigma x gbarA +1.10, tauA x delta_gA -0.75.
- **Weak axes.** Below 0.05 on y and 0.02 on the bursting share: x0, U_0_sr, U_0_ar, Omega_f_ar, Omega_f_sr, U_max.
- **The box leans active.** Interior: silent 1.0 %, bursting 44.8 %, above 20 Hz 49.2 %. Corners: silent 23.0 %, the highest at 328 Hz (98 % of 1/t_ref).
- **For SBI** `[OWN]`:
  - Features built on rate and burst share will leave the six release and facilitation axes near their prior. If those axes matter, the features must see release dynamics (within-burst ISIs, burst shape).
  - Expect saturated runs near 1/t_ref and silent runs; screen them before training.

## 7. Open items and commands

- **Launch, code, U_0_sr box.** Confirm them from the campaign itself:

```bash
cd ~/ANN/Astro-Neuron-Network/hpc/Phenomenological_finalv1/LindaColloDUP && head -40 campaign_lcdup15_rho1300_neur_n01/PROVENANCE_*.txt
python -c "import json,glob; f=sorted(glob.glob('campaign_lcdup15_rho1300_neur_n01/sweep_*/job_args.json'))[0]; d=json.load(open(f)); a=d['_axis_declaration']['swept_axes']; print(f); print(len(d['_active_param_names']), {k: (v['low'], v['high'], v['log']) for k, v in a.items()})"
python -c "import numpy as np,glob; f=sorted(glob.glob('campaign_lcdup15_rho1300_neur_n01/sweep_*/topo_00000/iter_00000.npz'))[0]; z=np.load(f); print(f, sorted(z.files), 'seed_verified' in z.files)"
```

  Expected for the code described here: 13 active names; 14 swept axes, among them U_0_sr (0.1, 1.0, True) and conn_prob (0.01, 0.4, False); `seed_verified` False.

- **Progress and completion.**

```bash
python aggregate_sweep.py campaign_lcdup15_rho1300_neur_n01 --target 1000000 | tail -3
qstat -u $USER | grep lcdup
```

- **The seed decision** (section 5).
- **Observation.** No observation x_o is assigned to this campaign. The decision log's D-001 binds the GiuliaGate simulations and the A50 cultures (`docs/project/ANN_decisions_and_ideas_log.md`) `[REPO]`. **[corrected 2026-11-17]** The SBI project now pairs n01 with the DUP15HD recordings (35 wells), per its request of 2026-11-17 (SBI decisions D-138 to D-141); no ANN decision records it.
