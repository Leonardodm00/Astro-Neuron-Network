# Reply to the SBI request of 2026-11-17: lcdup15 n01 for LindDUP

| Date | Change |
|---|---|
| 2026-11-17 | Created in reply to `REQUEST_2026-11-17_from_SBI_LindDUP_to_ANN.md` (SBI results-commentary chat, via the user). Built from the repository at `main` 1789e40 and at f216b00 (the launched code), sandbox runs of that code, project knowledge and past chats. Nothing was run on davinci for it. |
| 2026-11-20 | Sections 0, 2.2 and 4.5 corrected (decision D-005 of the ANN log): a topology counts only when all its runs left a sidecar; failure records no longer complete it. Evidence: the inspection of n01 on 2026-11-20 (ANN SPEC Block 14), where the records of three tasks sat in topologies that reached k by count. The audit counts sidecars only and lists each partial topology's counts ([RUN] on a fake campaign). "At most one partial topology per cut task" is not borne out (22 found where 16 tasks stopped short). |

Basis tags, in the request's terms: `[REPO]` read in the repository; `[RUN]` run here on the repository's code at the stated commit, or on files built in the campaign's layout (sandbox: Python 3.13.18, numpy 2.5.3, brian2 2.11.0; davinci's environment not checked); `[KB]` project-knowledge snapshot; `[PUBMED-FT]` PubMed, full text read; `[OWN]` own inference; `[CLUSTER]` read in files the author brought back from davinci (added 2026-11-20); **NOT KNOWN** where no source has it.

Paths are relative to the campaign's launch folder on davinci, `~/ANN/Astro-Neuron-Network/hpc/Phenomenological_finalv1/LindaColloDUP/`; run the commands there after `conda activate brian_final` `[REPO]`. Line numbers refer to f216b00's unpatched files.

## 0. Read first: 192 or 48 simulations share each topology

`LindaColloDUP15_SBI_brief.md` said every simulation has its own topology. That was wrong; the brief is corrected (section 5 below). For each outer iteration, `HPC_main_sweep.main` draws one conn_prob and one topology, then `n_workers x n_params_per_worker` parameter vectors, and runs them all on that topology (lines 1648-1866). The launch passes `--n_workers` = the job's cores and `--n_params_per_worker 1` (`submit_LCDUP.sh` lines 128, 169-171; `launch_LCDUP.sh` lines 69-70) `[REPO]`.

| queue | topologies per task | simulations per topology | simulations per task |
|---|---|---|---|
| cfd, 192 cores | 32 | 192 | 6,144 |
| cpu, 48 cores | 32 | 48 | 1,536 |

The 192 or 48 theta of a topology are independent prior draws (lines 1789-1791). Everything else is shared: conn_prob, the wiring, the neuron positions, and the build seed 50, so also V(0) and the bias pattern (SPEC Block 11) `[REPO]`.

Consequences for the SBI side `[OWN]`:
- conn_prob has one value per topology. The number of distinct values is the number of topologies (printed by the audit in 2.2), not the number of simulations.
- Split the training, validation and calibration (SBC, coverage) sets by topology folder (`sweep_*/topo_*`). A per-simulation split puts sibling runs, with the same network and the same start of the noise stream, on both sides.
- A topology cut short by a kill holds a biased subset of its theta (2.2). **[corrected 2026-11-20]** So may a topology in which a run left a failure record instead of a sidecar: drop every topology whose runs did not all leave a sidecar (2.2, "Failure records").

## 1. Well-to-donor map (item 1): NOT KNOWN

The ANN project has no record of the DUP15HD wells, their donors, lines or platings. What was searched:
- the repository, with `git grep` for DUP15, DATA_C, DATA_P and donor;
- project knowledge;
- past chats (six queries) `[REPO, KB]`;
- PubMed, two queries for Dup15q (15q11-q13 duplication) iPSC-derived neurons on MEAs: no hits, so no publication describing the cohort was found;
- not the data repositories (Synapse, DANDI, ...): no repository connector is available in this session.

The only DUP15HD facts in the ANN repository are the virtual-MEA probe preset (4.3).

The source is the experimental side's plating records. Ask for one row per well with these levels, since wells can share any of them `[OWN]`:

| level | why A5 needs it |
|---|---|
| donor (individual) | the unit A5 pairs on |
| line / clone | one donor can give several clones. In an isogenic design (a mosaic donor, or a gene-edited pair), a control line and a patient line come from the same individual, so one donor sits in both classes. Such a pair is described in the Kleefstra MEA study in the project files `[KB: Nat Commun 2019, 10:4928, doi 10.1038/s41467-019-12947-3, full text]` |
| differentiation / plating date | wells of one plating are technical replicates of one culture batch |
| plate, DIV at recording | wells recorded at different ages are not comparable |

If no two wells share a donor, A5 has no pairs on this cohort.

## 2. After the arrays are removed (item 2)

### 2.1 Final count: NOT KNOWN here

After the `qdel`, check that no lcdup job remains, then count:

```bash
qstat -u $USER | grep lcdup ; python aggregate_sweep.py campaign_lcdup15_rho1300_neur_n01 --target 1000000 | tail -3
```

- `done` counts the `topo_*/iter_*.json` sidecars `[REPO, aggregate_sweep.py lines 168-174]`. By the write order (2.2), every sidecar has a complete npz beside it.
- `discarded` counts the lines of `topo_*/_failures.jsonl` `[REPO]`.

### 2.2 What a kill can leave, and what each tool does with it

Write order for each topology `[REPO, HPC_main_sweep.py]`:

| step | file | how it is written | line |
|---|---|---|---|
| 1 | `topo_<NNNNN>/` | `os.makedirs` | 1651 |
| 2 | `topology.npz` | `np.savez_compressed`, straight to the final name | 1736 |
| 3 | `spatial_layout*.png` | matplotlib | 1737 |
| 4 | `topology_meta.json` | atomic: `.tmp`, then `os.replace` | 1782 |
| 5 | each simulation, once the topology is dispatched: `iter_<NNNNN>.npz` | `np.savez_compressed`, straight to the final name | 1224 |
| 6 | then `iter_<NNNNN>.json` | atomic | 1303 |

So a kill can leave:

| left behind | killed during | content |
|---|---|---|
| topology folder without `topology_meta.json` | steps 1-4 | no simulation; `topology.npz` missing, truncated or complete |
| `iter_*.npz` without its `.json` | step 5, or between 5 and 6 | a truncated npz, or a complete orphan |
| `iter_*.json.tmp` | step 6 | the npz beside it is complete |
| partial topology: sidecars + failures fewer than 192 or 48 **[corrected 2026-11-20: sidecars fewer than 192 or 48; failure records do not count, see "Failure records" below]** | steps 5-6 | only the runs that finished first. Run time grows with activity, so they lean toward quiet theta `[OWN]` |

A task runs its topologies one after another, so each task cut by a kill (this qdel, or an earlier 24 h walltime) has at most one partial topology `[REPO]`. **[corrected 2026-11-20]** Not borne out on n01: the inspection of 2026-11-20 (ANN SPEC Block 14) counted 22 topologies with sidecars + failure records below k, where 16 tasks stopped short of 32 topologies' worth of runs `[CLUSTER, the author's zip: inspect_summary.json, aggregate_index.log]`. Which folders make the difference is not known yet; the corrected audit below lists every partial topology with its counts.

**Failure records** **[added 2026-11-20, decision D-005 of the ANN log]**. A run whose `device.run` raises leaves a line in `_failures.jsonl` instead of a sidecar (the per-run `except Exception`, lines 1113-1212) `[REPO]`. On n01 the 1,381 records lie in 12 cfd tasks, none in the other 177 `[CLUSTER, aggregate_index.log]`. In three of them (0013, 0028, 0036), sidecars + records make a whole number of topologies, and their 241 records are exactly those the inspection found in topologies the first rule called complete `[OWN, arithmetic on the two files]`. What made these runs raise is not established: the kill, a resource failure of the slow runs, or theta itself (the non-finite spike check). In the first two cases the sidecars beside the records are a selected subset of the topology's theta, the first finishers or the runs that were not the heaviest; only in the third are they not `[OWN]`. Hence the rule: a topology counts only when every one of its 192 or 48 runs left a sidecar. If theta was the cause, this drops about 335 usable runs of n01 (0.07 %) `[OWN]`. The records' `error_type`, `error_msg` and `timestamp` settle the cause.

How the tools treat these files:
- **`aggregate_sweep.py`**, with `--target` or `--write-index` `[REPO; RUN, f216b00's copy on the fake campaign of the audit below: it indexed the 7 simulations with sidecars, including the 2 of the partial topology, and skipped the orphan npz, the .json.tmp and the topology without metadata]`:
  - it counts and indexes sidecars only, and opens no npz unless `--burst-metrics` is given;
  - it skips an unreadable sidecar with a `! skipped` line;
  - so it ignores all the leftovers above except partial topologies, which it indexes;
  - `--burst-metrics` opens only indexed npz. Its `except` clause (OSError, KeyError, ValueError) would not catch a truncated file, but indexed npz are complete by the write order. `[RUN]`: a truncated npz raises `zipfile.BadZipFile`, a 0-byte one `EOFError`.
- **`hpc/MEA Traces/process_campaign.py`** (the virtual MEA, 4.3) globs `iter_*.npz` `[RUN, on a fake task at main 1789e40]`:
  - a truncated npz is logged in `<out>/topo_*/_failures.log` and skipped;
  - a complete orphan is processed, so it appears in the MEA output but not in `campaign_index/`; join the two on (task, topo_idx, iter_idx);
  - a topology folder without a readable `topology.npz` stops the run of that whole sweep task: FileNotFoundError or BadZipFile, exit 1, no `mea_manifest.json`.

Rules `[OWN]`:
- A simulation is usable if and only if its `iter_*.json` exists.
- A topology is usable if and only if its `topology_meta.json` exists.
- Drop partial topologies whole. Removing a whole block of independent prior draws biases nothing; keeping only the runs that finished first does (4.5). **[corrected 2026-11-20]** Partial: fewer sidecars than 192 or 48, failure records not counted ("Failure records" above).

Audit. It only reads, except for the list it writes to `artifacts/`, which git ignores:

```bash
mkdir -p artifacts && python -c "import glob,json,os; C='campaign_lcdup15_rho1300_neur_n01'; T=sorted(glob.glob(C+'/sweep_*/topo_*')); K={d: (lambda a: a['_resolved_n_workers']*a['n_params_per_worker'])(json.load(open(d+'/job_args.json'))) for d in sorted(glob.glob(C+'/sweep_*'))}; L={t: set(os.listdir(t)) for t in T}; J=lambda f: {x[:-5] for x in f if x.startswith('iter_') and x.endswith('.json')}; N=lambda f: {x[:-4] for x in f if x.startswith('iter_') and x.endswith('.npz')}; F=lambda t: sum(1 for l in open(t+'/_failures.jsonl') if l.strip()) if '_failures.jsonl' in L[t] else 0; P=[t for t in T if 'topology_meta.json' in L[t] and len(J(L[t])) < K[os.path.dirname(t)]]; open('artifacts/lcdup15_n01_partial_topologies.txt', 'w').write(''.join('%s %d %d %d\n' % (os.path.relpath(t, C), K[os.path.dirname(t)], len(J(L[t])), F(t)) for t in P)); print('topology folders', len(T), '| without topology_meta.json', sum('topology_meta.json' not in L[t] for t in T), '| iter npz without json', sum(len(N(L[t])-J(L[t])) for t in T), '| json.tmp left', sum(sum(x.endswith('.json.tmp') for x in L[t]) for t in T), '| partial topologies', len(P), 'holding', sum(len(J(L[t])) for t in P), 'sims and', sum(F(t) for t in P), 'failure records (listed in artifacts/lcdup15_n01_partial_topologies.txt: folder, k, sidecars, records)')"
```

- `[RUN]` on a fake campaign with one of each defect: each was counted once, and the partial topology was listed. **[corrected 2026-11-20]** The command counted sidecars + failure records against k (`len(J(L[t]))+F(t) < K[...]`); it now counts sidecars only, and its list gives each partial topology's k, sidecars and records. `[RUN]` on a fake campaign of two tasks (k = 4 and 2) with a complete topology, two topologies with failure records, one with an orphan npz and a `.json.tmp`, one with no run and one without metadata: the 4 partial topologies listed with their counts, the other defects counted as expected.
- Expected on n01 `[OWN]`: one partial topology per task cut by a walltime kill or by the qdel, and few or no other leftovers. **[corrected 2026-11-20]** The inspection counted 22 by the first rule; by the corrected rule expect 25 if one topology per task holds the 241 records above `[OWN]`.
- A topology folder without `topology_meta.json` holds no simulation. Before running `process_campaign.py` on its task, move it out of the sweep folder. That is your call; nothing else reads the folder.

### 2.3 `campaign_index/`: NOT KNOWN whether written

The project has no record of a `--write-index` run on n01.

When written, the index is `campaign_lcdup15_rho1300_neur_n01/campaign_index/` `[REPO, aggregate_sweep.py lines 274-292]`:
- `theta_all.npy`: N x 37, float64;
- `npz_paths.txt`: absolute paths, row-aligned with it;
- `summary.csv`.

A copy written before the qdel is stale. Check whether one exists, rewrite it, then check that every indexed npz is a complete zip (this reads only the end of each file):

```bash
ls -la campaign_lcdup15_rho1300_neur_n01/campaign_index/ 2>&1 | head
python aggregate_sweep.py campaign_lcdup15_rho1300_neur_n01 --write-index | tail -3
python -c "import zipfile; P=open('campaign_lcdup15_rho1300_neur_n01/campaign_index/npz_paths.txt').read().splitlines(); B=[p for p in P if not zipfile.is_zipfile(p)]; print('indexed npz', len(P), '| not a complete zip:', len(B), B[:5])"
```

- The last check should print 0 `[OWN, from the write order]`. `[RUN]`: `is_zipfile` returned False for an npz cut to 0 bytes, to half its length and to all but its last 22 bytes, and True for the whole file.
- `--write-index` reads only the sidecars, about 4.4e5 small JSON files. `--burst-metrics` reads every npz, so run it as a job `[OWN]`.
- The docstring of `aggregate_sweep.py` still describes an older campaign (N x 35, 22 free axes). The code itself is not campaign-specific `[REPO]`.

## 3. Code that produced n01 (item 3)

`PROVENANCE_<launch id>.txt` holds no commit hash `[REPO, launch_LCDUP.sh lines 306-331]`. Its fields:
- campaign tag, launch id, repository folder, seed range, mode and group;
- the freeze list, `v_clamp_tref`, the `dV/dt` line;
- density and well size, simtime, conn_prob box, g boxes, sizing, queues, job ids;
- the sha256 of the seven files below, which are the folder's patched copies.

It can only be read on davinci.

Reference values `[RUN]`: the folder rebuilt at f216b00 in the sandbox, with each step's output as expected:
- step 0: the copy;
- step 4: the freeze and box patches, `PATCHED` twice, then `RESULT: 41 passed, 0 failed`;
- step 5a: `ALREADY CLAMPED`.

| file | sha256 at f216b00 after steps 0-5a | the same bytes for a step 0 run between (+01:00) |
|---|---|---|
| `HPC_main_sweep.py` (patched) | `7288893ea64fe34ed662b1b4906c1c7465cd67f97e717975bfb89d4856ceb755` | 2026-10-31 11:10 and 2026-11-03 09:32 |
| `HPC_single_run.py` | `f71851d52ac00a987406acdb34747f51a1d7a602c8ef16ede3beedf0dd18dfee` | 2026-10-31 06:42 and 2026-11-04 10:01 |
| `ASD_fun_BD_cpp.py` | `028e22c43e3c51367dbfd40010d3117688b95e292cd53ae3f47ad3a6da9dfcc5` | 2026-10-31 11:10 and 2026-11-04 13:24 |
| `synapse_pdist.csv` | `2845430bbccc1bcc74da815c78ffcc1f5bf28300c38281a47efd08b8c18efd0e` | since 2026-10-31 06:42 |
| `launch_LCDUP.sh` | `6a373e4301e0e0b95458534bea0a0f9a5af68657e4dd7140f2381009dcc0ead1` | f216b00 (2026-11-02 10:02) and 2026-11-04 15:55 |
| `submit_LCDUP.sh` | `99e204626371b5399bccd96717e21cf96aa13958b8c1a9504b800db82d82dae2` | f216b00 and 2026-11-04 15:55 |
| `seed_alloc.py` | `fc3d16b58f150ae5054b1bd5f1bba1926689be22df0ca2cc62efad1a093cfa46` | since 2026-10-30 19:11 |

Compare on davinci. The command prints `same as f216b00` or `DIFFERS` for each file:

```bash
python -c "import glob,re; R={'HPC_main_sweep.py':'7288893ea64fe34e','HPC_single_run.py':'f71851d52ac00a98','ASD_fun_BD_cpp.py':'028e22c43e3c5136','synapse_pdist.csv':'2845430bbccc1bcc','launch_LCDUP.sh':'6a373e4301e0e0b9','submit_LCDUP.sh':'99e204626371b539','seed_alloc.py':'fc3d16b58f150ae5'}; P=sorted(glob.glob('campaign_lcdup15_rho1300_neur_n01/PROVENANCE_*.txt')); print(P); H={m.group(2):m.group(1) for m in re.finditer(r'^([0-9a-f]{64})\s+(\S+)\s*$', open(P[0]).read(), re.M)}; [print('%-18s %-16s %s' % (k, H.get(k, '(absent)')[:16], 'same as f216b00' if H.get(k, '').startswith(v) else 'DIFFERS')) for k, v in R.items()]"
```

`[RUN]` on a PROVENANCE file built from the rebuilt folder, with one hash altered: that file printed `DIFFERS` and the other six matched. If all seven match, n01 ran the f216b00 pipeline. The SBI side already found conn_prob [0.01, 0.4] and U_0_sr [0.1, 1] in `job_args.json`; that points to f216b00's launcher and to a `HPC_main_sweep.py` older than 004ed7b `[OWN]`.

The three properties the brief inferred:

| property | in the launch-commit code | check on davinci, expected |
|---|---|---|
| 37-D vectors | `N_DIMS 37` `[RUN]` | `params (37,)`, `theta (37,)` |
| refractory clamp on | the dV/dt line ends `(unless refractory)`; t_ref 3 ms `[REPO, ASD_fun_BD_cpp.py lines 669, 1239]` | PROVENANCE: `v_clamp_tref : yes`, and its dV/dt line ends `(unless refractory)` |
| per-run seed not applied | `device.seed(seed_run)` after the build does nothing, so every run starts from the build seed 50. Found 2026-11-03 (SPEC section 8), fixed 2026-11-04 by Block 11 `[REPO]`. `seed_verified` occurs 0 times in f216b00's `HPC_main_sweep.py` and twice at main `[RUN]` | `seed_verified key: False`, `noise_mode fresh` (a label, not what ran), `seed_run` >= 0 |

```bash
python -c "import numpy as np,glob; f=sorted(glob.glob('campaign_lcdup15_rho1300_neur_n01/sweep_*/topo_00000/iter_00000.npz'))[0]; z=np.load(f); print(f); print('params', z['params'].shape, '| theta', z['theta'].shape, '| seed_verified key:', 'seed_verified' in z.files, '| conn_prob key:', 'conn_prob' in z.files, float(z['conn_prob']), '| noise_mode', str(z['noise_mode']), '| seed_run', int(z['seed_run']))"
```

`[RUN]` on a fake npz in this layout. A later step 0 may have replaced the folder's copies, so check the PROVENANCE file and the npz, not the files now in the folder `[OWN]`.

## 4. What the SBI tools must read (item 4)

### 4.1 The 13 free axes

`_active_param_names` lists the free axes in increasing registry index, which is their column order in the 37-D `theta` `[RUN]`. `_apply_freeze` returns the kept indices sorted: of the 23 axes of the `neuron_synapse` group, 10 are frozen. `job_args.json` also stores the indices as `_active_indices`. Read them from there and check that they are equal in every task, rather than hard-coding them.

Coordinates `[REPO]`:
- log axes: theta_j = ln(v_j / u_j), with v_j the value and u_j the unit in the table;
- the linear axis (Sigma): theta_j = v_j / u_j.

| column | name | unit u_j | box in u_j | scale | box in theta |
|---|---|---|---|---|---|
| 0 | Sigma | mV | [3, 15] | lin | [3, 15] |
| 1 | gbarA | nS | [0.01, 10] | ln | [-4.60517, 2.30259] |
| 4 | tauA | ms | [10, 2000] | ln | [2.30259, 7.60090] |
| 5 | U_0_ar | 1 | [1e-4, 0.05] | ln | [-9.21034, -2.99573] |
| 6 | U_max | 1/ms | [0.1, 1] | ln | [-2.30259, 0] |
| 7 | U_0_sr | 1 | [0.1, 1] | ln | [-2.30259, 0] |
| 8 | Omega_f_sr | 1/s | [0.1, 4.5] | ln | [-2.30259, 1.50408] |
| 9 | Omega_f_ar | 1/s | [0.1, 4.5] | ln | [-2.30259, 1.50408] |
| 10 | Omega_d | 1/s | [0.1, 4.5] | ln | [-2.30259, 1.50408] |
| 14 | g_ampa | nS | [0.01, 10] | ln | [-4.60517, 2.30259] |
| 15 | g_nmda | nS | [0.01, 10] | ln | [-4.60517, 2.30259] |
| 16 | delta_gA | nS | [0.005, 10] | ln | [-5.29832, 2.30259] |
| 17 | x0 | 1 | [0.05, 0.5] | ln | [-2.99573, -0.69315] |

`[RUN]`: the patched f216b00 module printed these boxes.
- The other 24 columns are constant `[REPO]`.
- Do not evaluate the registry's 37-D box on `theta_all` rows: I_inj is held at 4.0 pA, outside its registry box [4.5, 7.5] pA `[RUN]`.

### 4.2 conn_prob

- Not a column of theta: `PARAM_NAMES` has 37 names, none of them conn_prob `[RUN]`.
- Stored per simulation as the npz key `conn_prob`, float64, written only under the flat rule (`conn_rule` 'flat'). Also in the sidecar, `topology_meta.json` and `summary.csv` `[REPO]`.
- Drawn once per topology, uniform in natural units on [0.01, 0.4] (`master_rng.uniform`, line 1666), so its coordinate is linear `[REPO]`.
- Shared by the 192 or 48 simulations of its topology (section 0).

### 4.3 Time base, bins, electrode geometry

**Spike times** `[REPO]`:
- `spk_N_t`: float32, seconds from the start of the run; integration step dt = 0.05 ms; 180 s per run.
- `spk_N_i`: int32, neuron index 0-158.
- The range-check analysis discards the first 2 s as the start-up transient (`t_skip` 2.0 in `hpc/range_check/range_analysis.py`).

**Bin.** The ANN side prescribes none for SBI; it must be the bin used on the recordings `[OWN]`. The only population-rate bin in ANN code belongs to `burst_metrics.py`'s own network-burst detector on the full raster: a 5 ms grid with 15 ms Gaussian smoothing `[REPO]`.

**Electrode geometry.** None is stored with a simulation `[REPO; RUN, positions from f216b00's build_topology]`:
- `topology.npz` holds `N_pos`: 159 x 2, in um, inside [0, 350]^2;
- `topology_meta.json` holds `c_max` 350.

**The virtual MEA already in the repository.** `hpc/MEA Traces/process_campaign.py` turns spike times and positions into extracellular traces, then re-detects spikes: band-pass 300-3000 Hz, threshold 5 x robust sigma, 2 ms dead time `[REPO; KB, MEA_Analysis_Reference.md]`.
- Its default geometry is the DUP15HD preset of smoke test W1 in `smoke_test_mea_workers.py`: `--n_side 3 --pitch 60 --edge 25 --fs 10110.09`. That is square electrodes of 25 um side, centred on the culture `[REPO]`.
- 10110.09 Hz is the sample rate the user specified when the pipeline was written (`THEORY.md`) `[REPO]`. Check it, and the electrode edge, against the recordings' metadata `[OWN]`.
- Output `[REPO]`: `mea_iter_*.npz`, with `det_ch`, `det_t` (s), `det_amp` (uV), `electrode_centers` (um), `params`, `theta`, `conn_prob`, `topo_idx`, `iter_idx`, `seed_run`.
- The recording noise is seeded per simulation from `seed_run` (`--noise_seed_scheme sim`, labelled D-072 in the code's comments) `[REPO]`.
- Its `simtime` key is the last spike time rounded up (line 267), and the trace stops there; it is 1 s for a run with no spike. Take the duration, 180 s, from `topology_meta.json` (`simtime_s`) instead `[REPO]`.

**What the probe sees on an n01 network** `[RUN]`. Twenty topologies from f216b00's `build_topology`, default scaling (75 uV at 30 um, 1/r^2, threshold 25 uV):
- the electrodes sit at 115, 175 and 235 um on each axis;
- every neuron lies within the reach radius, 164.3 um;
- 55 to 67 of the 159 neurons (mean 59.5) are within detection range of at least one electrode.

In busy networks the robust noise estimate rises, so fewer neurons are detected `[OWN]`.

`hpc/Electrode Traces Extractor/` also exists in the repository; it was not checked.

### 4.4 Silent and saturated runs: NOT KNOWN for the campaign

`summary.csv` has the columns needed `[REPO]`:
- `mean_FR_Hz`;
- `frac_active`, the share of neurons above 0.1 Hz over the 180 s;
- `n_neuronal_spikes`.

The shares have not been computed. After 2.3:

```bash
python -c "import csv,statistics as S; it=csv.reader(open('campaign_lcdup15_rho1300_neur_n01/campaign_index/summary.csv')); h=next(it); i,j,k=h.index('mean_FR_Hz'),h.index('frac_active'),h.index('n_neuronal_spikes'); D=[(float(r[i]),float(r[j]),int(float(r[k]))) for r in it]; n=len(D); p=lambda c: '%.2f %%' % (100.0*c/n); print('runs', n, '| no spike', p(sum(d[2]==0 for d in D)), '| silent (frac_active < 0.1)', p(sum(d[1]<0.1 for d in D)), '| mean rate > 20 Hz', p(sum(d[0]>20 for d in D)), '| >= 300 Hz (0.9/t_ref)', p(sum(d[0]>=300 for d in D))); q=S.quantiles([d[0] for d in D], n=100); print('mean rate (Hz) percentiles 1 5 25 50 75 95 99:', [round(q[m-1], 3) for m in (1,5,25,50,75,95,99)])"
```

- `[RUN]` on a 1,000-row `summary.csv` with known shares: it printed them exactly.
- Definitions: "silent" follows the range check (frac_active < 0.1). "Saturated" here is mean_FR_Hz >= 300 Hz, that is 0.9/t_ref; that threshold is my choice `[OWN]`.
- The only shares measured on this setup come from the range check (brief, section 6) `[KB]`:
  - interior: silent 1.0 %, above 20 Hz 49.2 %;
  - corners: silent 23.0 %, highest rate 328 Hz.

  These do not stand for the campaign: the range check used a grid design rather than prior draws, conn_prob up to 0.5, and 20 s runs.

### 4.5 Screening before training `[OWN, with PUBMED-FT]`

Rule: screen on the observation side only, never on theta, with a statistic computed the same way on simulations and recordings.

Notation:
- theta in Theta: the inferred parameters, with prior p(theta);
- x in X: the observation the posterior is conditioned on (the SBI features of the re-detected 3 x 3 activity);
- p(x | theta): the simulator's distribution of x, for each fixed theta;
- A, a subset of X: the kept observations, with indicator 1_A(x) and P(x in A) > 0;
- p(x) = integral over Theta of p(theta') p(x | theta') d theta': the marginal density of x; theta' and x' below are integration variables.

Training on the kept pairs only means training on

$$p_A(\theta, x) \;=\; \frac{p(\theta)\, p(x \mid \theta)\, \mathbb{1}_A(x)}{\int_\Theta \int_X p(\theta')\, p(x' \mid \theta')\, \mathbb{1}_A(x')\, dx'\, d\theta'} . \tag{1}$$

The kept pairs follow the prior-times-likelihood joint, cut down to A and renormalised. For each fixed x in A with p(x) > 0, the indicator equals 1 and the normaliser cancels:

$$p_A(\theta \mid x) \;=\; \frac{p(\theta)\, p(x \mid \theta)}{\int_\Theta p(\theta')\, p(x \mid \theta')\, d\theta'} \;=\; p(\theta \mid x). \tag{2}$$

At any observation inside A, a network trained on the screened pairs targets the unscreened posterior. Only the marginal of theta in the training set changes, to p_A(theta) proportional to p(theta) P(x in A | theta). Consequences:
- **Choose A from statistics of x.** Examples: no detected event on any of the 9 electrodes after the first 2 s; a detected rate above a ceiling. Set the bounds so every recorded well passes with margin, and apply the same rule to the recordings. A well outside A has no posterior from this network.
- **Do not screen on a quantity that is not a function of x**, such as the full-raster `mean_FR_Hz`. Call that quantity s(y), with y the raw output. A screen on it makes the network target p(theta | x, s(y) in A), which equals p(theta | x) only if P(s(y) in A | theta, x) does not depend on theta. Use `summary.csv` to size the problem, not to screen.
- **Drop partial topologies whole** (2.2). Their selection is on run time, which depends on y, not on x alone. **[corrected 2026-11-20]** Partial now includes a topology whose runs left failure records (2.2, "Failure records").
- **Apply the same screen to the held-out simulations** used for SBC and coverage, and report the fraction removed.

Equation (2) is my derivation `[OWN]`. According to PubMed (full text read), Deistler, Macke and Gonçalves (2022) [doi:10.1073/pnas.2207632119](https://doi.org/10.1073/pnas.2207632119) handle this as follows `[PUBMED-FT]`:
- they discard simulations whose summary features are undefined (no bursts) before training SNPE;
- they add a classifier that rejects parameters predicted to give invalid output; the accepted parameters form a "restricted prior";
- they state that the procedure converges to the correct posterior. The proof is not in the main text read here.

## 5. Corrections made to `LindaColloDUP15_SBI_brief.md` on 2026-11-17

- Sections 1, 2 and 5: 192 (cfd) or 48 (cpu) simulations per topology, not one; conn_prob and the topology are per topology.
- Section 4: a run is also discarded when it raises (a binary crash, a Brian2 error), not only when its spike times are non-finite (lines 1113-1212) `[REPO]`.
- Section 5: what the shared topology means for the noise streams (SPEC section 8).
- Section 7: the SBI pairing with DUP15HD.

## 6. Order on davinci after the qdel

1. `qstat -u $USER | grep lcdup` prints nothing.
2. Count (2.1).
3. Audit (2.2). Move topology folders without `topology_meta.json` aside if the virtual MEA is to run.
4. Provenance and the three properties (3).
5. Index and completeness (2.3).
6. Shares (4.4).

Send the output of steps 2-6 to the SBI chat.
