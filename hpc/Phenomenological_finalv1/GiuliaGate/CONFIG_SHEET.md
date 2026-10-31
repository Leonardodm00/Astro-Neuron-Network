# Giulia gate campaign: configuration to decide

Proposal of 2026-10-31. No code reads these files yet: the launch package (as for LindaColloDUP) is built from them once you have decided. Decide by editing `prior_config.tsv` (columns status, low, high, value) and `campaign_config.tsv` (column value), or by answering the decisions at the end in chat.

## The setup

Giulia's design, as in her campaigns v1-v5: MODE Full (neurons and astrocytes), the neuron parameters fixed, the synaptic and astrocytic parameters drawn from the prior (her group `synapse_astro`, 22 axes). Two changes: the astrocyte contact gate, a new per-network axis, and the refractory hold (V held at VR during t_ref). It runs from the clone, on the 37-D default pipeline (`specs/SPEC.md` Block 8), in a folder of its own. Plainly: the same experiment as v1-v5, except that each simulated culture now also draws what fraction of its synapses an astrocyte touches, and a neuron can no longer run away to NaN during its refractory period.

Where the values come from:
- the code of the two repository copies: the root 37-D default, and `hpc/Phenomenological_finalv1/Giulia/`, the copy of Giulia's setup (its README says it matches the cluster's `Giulia_Astro/`, not re-checked);
- Gorski et al. 2021, Table 1 (project knowledge, full text);
- past chats: 2026-06-25, your decisions for Giulia's setup; 2026-08-31, your paste of the v5 worker (C_MAX 300 um, 1200 neurons and 1200 astrocytes per mm^2, so 108 + 108).
- v5's own `job_args.json` was not read: the "Giulia copy (as v5)" columns take the repository copy's boxes and values as v5's, and the command at the end prints v5's, to confirm them.

## Decision A: the cell (the values the neurons are fixed at)

| quantity | Giulia RS cell (proposed) | registry cell (MEAMA, LindaColloDUP) | note |
| --- | --- | --- | --- |
| Cm (pF) | 200 | 17.25 | tau_m = Cm/gL: 20 ms vs 15 ms |
| gL (nS) | 10 | 1.15 | frozen row of the registry |
| El (mV) | -55 | -58.2 | model-file constant; Gorski 2021: -60, raised to -55 in the Giulia copy |
| VT (mV) | -50 | -51 | frozen row of the registry |
| VR (mV) | -55 | -58.2 |  |
| VA (mV) | -50 | -48 |  |
| DeltaA (mV) | 5 | 5 |  |
| gbarA (nS) | 10 | 0.9 |  |
| delta_gA (nS) | 1 | 0.135 |  |
| tauA (ms) | 200 | 200 |  |
| DeltaT (mV) | 2 | 2 | frozen row of the registry |
| I_inj (pA) | 40 | 4.0 (LindaColloDUP) | bias I_inj*u, u ~ U[-0.5, 0.5]: about +-2 mV at gL in both cells; the June plan said 7.5 pA |
| t_ref (ms) | 5 | 3 | model-file constant |
| gl_ref (nS) | 10 | 1.15 | model-file constant (noise scaling, = gL) |
| EA, VD (mV) | -70, -40 | -70, -40 | model-file constants |

- Gorski et al. 2021, Table 1, "Adaptive spiking": Cm 200 pF, EA -70 mV, EL -60 mV, VA -50 mV, DeltaA 5 mV, VR -55 mV, VT -50 mV, delta_gA 1 nS, gbarA 10 nS, gL 10 nS, tauA 200 ms (project knowledge, full text). The Giulia RS cell is that row with EL at -55 mV. Its t_ref 5 ms, VD -40 mV and DeltaT 2 mV are cited to the same paper in code comments, not checked here. I_inj is a bias width, not the paper's step current.
- The registry cell is the small cell of MEAMA and LindaColloDUP (its values are grounded in code comments to human iPSC-derived neurons; not re-checked here).
- Cost in code. The registry cell needs only the freeze list. The Giulia RS cell needs one patch more: El, t_ref and gl_ref are constants of the model file, and VT and gL are frozen rows of the registry, which the freeze list cannot set. It would be a patcher with its test, like the box patch of LindaColloDUP.
- Synaptic scale (own reasoning, not measured). The same g_ampa, g_nmda in nS act against a leak 8.7 times larger in the RS cell (10 vs 1.15 nS), so the g boxes go with the cell: v1-v5 used [0.05, 5] nS with the RS cell, LindaColloDUP uses [0.01, 10] nS with the registry cell. The range check measures it.

## Free axes: 22 per run, 2 per network

| axis | unit | proposed box | scale | Giulia copy (as v5) | registry (MEAMA, LindaColloDUP) | meaning |
| --- | --- | --- | --- | --- | --- | --- |
| Sigma | mV | [2, 10] | lin | free [2, 10] lin | nominal 4; box [3, 15] lin | membrane noise amplitude (diffusion noise, sigma_V); free in Giulia v1-v5 although a neuron axis: decide (fix at 4 mV, or keep free) |
| U_0_ar | 1 | [0.0001, 0.05] | log | free [0.0001, 0.05] log | nominal 0.003; box [0.0001, 0.05] log | asynchronous release: per-spike increment of the release rate toward U_max |
| U_max | 1/ms | [0.1, 1] | log | free [0.1, 1] log | nominal 0.5; box [0.1, 1] log | asynchronous release: ceiling of the per-vesicle release rate |
| U_0_sr | 1 | [0.1, 1] | log | free [0.1, 1] log | nominal 0.15; box [0.1, 1] log | synchronous release probability (baseline) |
| Omega_f_sr | 1/s | [0.1, 4.5] | log | free [0.1, 4.5] log | nominal 2; box [0.1, 4.5] log | facilitation decay rate, synchronous release |
| Omega_f_ar | 1/s | [0.1, 4.5] | log | free [0.1, 4.5] log | nominal 1.42857; box [0.1, 4.5] log | decay rate of the asynchronous release rate |
| Omega_d | 1/s | [0.1, 4.5] | log | free [0.1, 4.5] log | nominal 2; box [0.1, 4.5] log | recovery rate of the vesicle pool (depression) |
| alpha_syn | 1 | [0.1, 1] | log | free [0.1, 1] log | nominal 1; box [0.1, 1] log | effect of gliotransmission on release probability (Full mode) |
| g_ampa | nS | [0.05, 5] | log | free [0.05, 5] log | nominal 0.3; box [0.01, 5] log | AMPA conductance |
| g_nmda | nS | [0.05, 5] | log | free [0.05, 5] log | nominal 0.3; box [0.01, 5] log | NMDA conductance |
| x0 | 1 | [0.05, 0.5] | log | free [0.05, 0.5] log | nominal 0.2; box [0.05, 0.5] log | quantal vesicle size (fraction of the pool per release) |
| O_G | 1/(uM*s) | [0.1, 10] | log | free [0.1, 10] log | nominal 1.5; box [0.1, 10] log | presynaptic receptor binding rate of gliotransmitter |
| Omega_G | 1/s | [0.001, 0.1] | log | free [0.001, 0.1] log | nominal 0.00833; box [0.001, 0.1] log | presynaptic receptor inactivation rate |
| O_beta | uM/s | [0.1, 5] | log | free [0.1, 5] log | nominal 1; box [0.1, 5] log | astrocyte IP3 production by PLCbeta (gain) |
| O_3K | uM/s | [1, 15] | log | free [1, 15] log | nominal 4.5; box [1, 15] log | astrocyte IP3 degradation by IP3-3K (rate) |
| Omega_5P | 1/s | [0.01, 1] | log | free [0.01, 1] log | nominal 0.1; box [0.01, 1] log | astrocyte IP3 degradation by IP3-5P (rate) |
| I_bias | uM | [0.3, 1] | lin | free [0.3, 1] lin | nominal 0.8; box [0.3, 1] lin | astrocyte exogenous IP3 set-point |
| F | uM/s | [0.1, 10] | log | free [0.1, 10] log | nominal 2; box [0.1, 10] log | permeability of the astrocyte IP3 flux through gap junctions and from the exogenous source |
| C_Theta | uM | [0.1, 2] | log | free [0.1, 2] log | nominal 0.5; box [0.1, 2] log | astrocyte Ca2+ threshold for gliotransmitter exocytosis |
| U_A | 1 | [0.1, 0.9] | lin | free [0.1, 0.9] lin | nominal 0.6; box [0.1, 0.9] lin | gliotransmitter release probability |
| G_T | mM | [50, 1000] | log | free [50, 1000] log | nominal 200; box [50, 1000] log | total gliotransmitter in the astrocyte |
| O_N | 1/(uM*s) | [0.03, 3] | log | free [0.03, 3] log | nominal 0.3; box [0.03, 3] log | astrocyte mGluR glutamate binding rate (neuron-to-astrocyte coupling) |
| conn_prob | 1 | [0.05, 0.6] | lin | free [0.05, 0.6] (June 2026 decision; v5 to confirm) | MEAMA [0.1, 0.6]; LindaColloDUP [0.01, 0.5] | flat rule: each directed neuron pair wired with this probability, one draw per network |
| stoa_gate_p | 1 | [0.2, 1.0] | lin | absent (every viable contact kept, p = 1) | MEAMA [0.2, 1.0] | contact gate: each viable contact (a synapse and its nearest astrocyte within stoa_cutoff) kept with this probability, one draw per network; uptake and gliotransmission gated together (Full mode only) |

Scale is computed from the box, as `HPC_main_sweep.py` does: log when low > 0 and high/low >= 10, otherwise linear. Moving a bound can flip it (Sigma becomes log if its low drops below 1 mV).

## Fixed axes: 15

| axis | unit | proposed value | Giulia copy (as v5) | registry | meaning |
| --- | --- | --- | --- | --- | --- |
| gbarA | nS | 10 | fixed 10 | nominal 0.9; box [0.01, 10] log | maximal subthreshold adaptation conductance; value of the chosen cell (here: Giulia RS) |
| delta_gA | nS | 1 | fixed 1 | nominal 0.135; box [0.005, 10] log | post-spike adaptation increment; value of the chosen cell (here: Giulia RS) |
| tauA | ms | 200 | fixed 200 | nominal 200; box [10, 2000] log | adaptation time constant; value of the chosen cell (here: Giulia RS) |
| VA | mV | -50 | fixed -50 | nominal -48; box [-73, -38] lin | half-activation voltage of subthreshold adaptation; value of the chosen cell (here: Giulia RS) |
| DeltaA | mV | 5 | fixed 5 | nominal 5; box [0.1, 15] log | slope of subthreshold adaptation; value of the chosen cell (here: Giulia RS) |
| VR | mV | -55 | fixed -55 | nominal -58.2; box [-68.2, -48.2] lin | reset potential; value of the chosen cell (here: Giulia RS) |
| I_inj | pA | 40 | fixed 40 | nominal 4; box [4.5, 7.5] lin | width of the per-neuron bias current: each neuron gets I_inj*u, u ~ U[-0.5, 0.5] fixed per network (mean 0); value of the chosen cell (here: Giulia RS) |
| Cm | pF | 200 | fixed 200 | nominal 17.25; box [9.2, 34.5] lin | membrane capacitance (tau_m = Cm/gL); value of the chosen cell (here: Giulia RS) |
| DeltaT | mV | 2 | fixed 2 | frozen 2 | spike-initiation slope; frozen in every group; value of the chosen cell (here: Giulia RS) |
| VT | mV | -50 | fixed -50 | frozen -51 | spike threshold; frozen in every group; value of the chosen cell (here: Giulia RS) |
| gL | nS | 10 | fixed 10 | frozen 1.15 | leak conductance; frozen in every group; value of the chosen cell (here: Giulia RS) |
| EC50_ampa | mmole | 8.6 | fixed 8.6 | nominal 8.6; box [2, 150] log | AMPA receptor half-activation glutamate concentration |
| EC50_nmda | mmole | 3 | fixed 3 | nominal 3; box [0.5, 50] log | NMDA receptor half-activation glutamate concentration |
| I_Theta | uM | 0.3 | fixed 0.3 | nominal 0.3; box [0.1, 1] log | IP3-gradient threshold of that flux (tanh) |
| omega_I | uM | 0.05 | fixed 0.05 | nominal 0.05; box [0.01, 0.5] log | steepness of that flux (tanh) |

## The two per-network axes

- conn_prob ~ U[0.05, 0.6], your June decision for this setup (v5 to confirm). With Nn = 108 the mean in-degree is 107 x conn_prob: 5.4 to 64 inputs (own arithmetic).
- stoa_gate_p ~ U[0.2, 1.0], as MEAMA. Each network draws p once; each viable contact (a synapse and its nearest astrocyte, if that astrocyte lies within `stoa_cutoff`, 70 um, of the bouton) is then kept with probability p, and glutamate uptake by the astrocyte and gliotransmission back to the synapse are gated together (`HPC_single_run.build_topology`, `ASD_fun_BD_cpp.nearest_astro_for_synapse`). In a sandbox culture every synapse was viable (`n_stoa=184/254 viable` with 254 synapses); at 1200 astrocytes per mm^2 that is expected to hold, so p is close to "the fraction of synapses an astrocyte touches" (own reasoning; each run prints kept/viable per network).
- What tissue electron microscopy reports, according to PubMed (full texts read in this session):
  - Genoud et al. 2006, mouse barrel cortex layer IV, serial-section TEM, spine synapses: any astrocytic contact 90.5 +- 1.6 % (control) and 94.9 +- 1.9 % (after 24 h of whisker stimulation); bouton-spine interface partly or fully surrounded 68.1 +- 1.7 % and 81.7 +- 3.5 % ([doi](https://doi.org/10.1371/journal.pbio.0040343)).
  - Ostroff et al. 2014, rat lateral amygdala, astrocytic membrane within 20 nm of the cleft perimeter: 45 +- 3.3 % of asymmetric spine synapses in naive rats, 34 +- 3.1 % after threat conditioning; shaft synapses about 30 % (asymmetric) and 6 % (symmetric) ([doi](https://doi.org/10.1002/cne.23523)).
  - Gavrilov et al. 2018, rat CA1, astrocytic leaflets within 50 nm: 72.7 % of thin and 83.5 % of mushroom spines ([doi](https://doi.org/10.3389/fncel.2018.00248)).
  - Ventura & Harris 1999 (rat CA1), cited for the box in `launch_campaign.sh`: full text not accessible (PMID 10436047), so no number from it.
  - So for spine synapses in tissue the fraction runs from about 0.34 to 0.95, depending on what counts as contact, the region and the condition; no value comes from cultures. The "0.68-0.82" of the launcher comment is Genoud's partly-or-fully-surrounded class, control vs stimulated, not a baseline range. [0.2, 1.0] covers the tissue range.

## Campaign settings

| key | proposed | unit | Giulia v5 | registry / MEAMA / LindaColloDUP | note |
| --- | --- | --- | --- | --- | --- |
| folder | hpc/Phenomenological_finalv1/GiuliaGate | - | - | - | proposed name; a folder of the clone, like LindaColloDUP/ |
| campaign_tag | cadex_hhgap_clamp_v1 | - | cadex_hhgap_v5 | - | decided 2026-10-30 for the rerun of Giulia simulations |
| mode | Full | - | Full | MEAMA Full; LindaColloDUP Neuronal | astrocytes on; the gate is read only in Full mode |
| cell | giulia_rs | - | giulia_rs | registry | giulia_rs = Gorski 2021 adaptive spiking, EL raised to -55; registry = the default cell |
| cell_El | -55 | mV | -55 | -58.2 | model-file constant, not a sweep axis |
| cell_t_ref | 5 | ms | 5 | 3 | model-file constant (refractory period, V held at VR) |
| cell_gl_ref | 10 | nS | 10 | 1.15 | model-file constant: leak used to scale the noise (= gL) |
| cell_EA | -70 | mV | -70 | -70 | model-file constant: adaptation reversal |
| cell_VD | -40 | mV | -40 | -40 | model-file constant: spike detection |
| density | 1200 | neurons/mm^2 | 1200 | MEAMA 800 (rho_total 1600); LindaColloDUP 1300 | - |
| density_astro | 1200 | astrocytes/mm^2 | 1200 | MEAMA 800 | Na = Nn (decided June 2026) |
| c_max | 300 | um | 300 | MEAMA 350; LindaColloDUP 350 | culture side; 300 gives Nn = Na = 108 (400 gives 192) |
| simtime | 200 | s | 200 (June 2026; v5 to confirm) | MEAMA 200; LindaColloDUP 180 | - |
| conn_rule | flat | - | flat | flat | box in prior_config.tsv (conn_prob) |
| conn_periodic | 0 | - | 0 | 0 | read only by the weibull rule |
| topology_mode | wallach | - | wallach | wallach | astrocyte contacts by joint Voronoi adjacency |
| displ_bias | 20 | um | 20 | 20 | - |
| gj_max_dist | 150 | um | 150 | 150 | wallach: cap on gap-junction distance |
| gj_dist | 200 | um | 200 | 200 | distance mode only (not read under wallach) |
| stoa_cutoff | 70 | um | 70 | 70 | a synapse has a viable contact when its nearest astrocyte lies within this distance of the bouton; the gate draws from these |
| stoa_sigma | 200 | um | 200 | 200 | distance mode only (not read under wallach) |
| refractory_clamp | yes | - | no (unclamped model) | yes | V held at VR during t_ref; the root model has it |
| k_params | 1 | - | v5 to confirm (n_params_per_worker) | MEAMA 1; LindaColloDUP 1 | parameter vectors per network per worker |
| seed_device_neuron_synapse_astro | 50/39/35/60 | - | 50/39/35/60 | 50/39/35/60 | fixed Brian2 seeds; launch seeds come from the seed guard |
| seed_stoa_gate | 71 | - | - | 71 | offset of the dedicated gate RNG |
| queues | intel:48:2 cpu:48:2 | - | - | MEAMA and LindaColloDUP | - |
| walltime | 24:00:00 | - | - | 24:00:00 | - |
| target | 1000000 | simulations | - | 1000000 | - |
| headroom_pct | 200 | % | - | 200 | - |
| n_topologies_worker | TBD | - | - | MEAMA from its measured time; LindaColloDUP 32 | sized from the first tasks (Full, 108+108, 200 s) |
| conda_env | brian_final | - | - | brian_final | - |

## Before the launch

- The standard prior range calibration on these boxes: 22 axes + conn_prob = 23 dimensions, so the corners are a resolution-8 fraction (16,384 runs + 512 centre points) and the interior 16,464 cells of the 7-bin grid (`claude/Prior_Range_Calibration_Standard.md`). The gate is drawn per network, not designed, unless you want it as a design dimension (code to add).
- `N_TOPOLOGIES_WORKER` is sized from the first tasks (Full, 108 + 108, 200 s).

## Decisions

1. The cell: Giulia RS (proposed) or the registry cell.
2. Sigma (noise): free in [2, 10] mV as in v1-v5 (proposed), or fixed at 4 mV.
3. I_inj, if Giulia RS: 40 pA (repository copy, proposed) or 7.5 pA (the June plan).
4. Synaptic axes: the 12 that Giulia's campaigns drew, free with the Giulia copy's boxes (proposed; g in [0.05, 5] nS), or other boxes, or some fixed.
5. Astrocytic axes: the 9 that Giulia's campaigns drew, free (O_N among them); I_Theta and omega_I fixed, as in those campaigns and in MEAMA.
6. conn_prob box: [0.05, 0.6] (proposed).
7. Gate box: [0.2, 1.0] (proposed); and whether it becomes a design dimension of the range check.
8. Culture: C_MAX 300 um, 108 + 108, as v5 (proposed), or 400 um, 192 + 192, the June plan; simtime 200 s.
9. Folder and tag: `hpc/Phenomenological_finalv1/GiuliaGate/` and `cadex_hhgap_clamp_v1` (proposed).

## Optional: v5's own values (reads data only)

```bash
cd ~/ANN/Phenomenological/Main/Giulia_Astro && python3 -c "import json,glob; f=sorted(glob.glob('campaign_cadex_hhgap_v5/sweep_*/job_args.json'))[0]; d=json.load(open(f)); print(f); print(' '.join('%s=%s' % (k, d.get(k)) for k in ('mode','sweep_group','Nn','Na','c_max','density','density_astro','simtime','conn_rule','conn_prob_lo','conn_prob_hi','topology_mode','displ_bias','gj_dist','gj_max_dist','stoa_cutoff','stoa_sigma','n_params_per_worker','n_topologies','seed_device','seed_neuron','seed_synapse','seed_astro'))); a=d.get('_axis_declaration',{}); [print('free ', k, v.get('low'), v.get('high'), 'log' if v.get('log') else 'lin') for k,v in sorted(a.get('swept_axes',{}).items())]; [print('fixed', k, v) for k,v in sorted(a.get('fixed_axes',{}).items())]"
```

It prints one line of settings, then a `free` line per drawn axis (box, scale) and a `fixed` line per fixed axis (value). Tested on a job_args.json of the root pipeline.
