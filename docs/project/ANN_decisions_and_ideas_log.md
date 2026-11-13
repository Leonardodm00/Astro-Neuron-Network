# SBI decisions and ideas log (Neuro-astro network)

| Date | Change |
|---|---|
| 2026-11-10 | Created at the path the project instructions name (it did not exist; until now campaign decisions were recorded in the campaign operations docs, e.g. `claude/GiuliaGate_Campaign_Operations.md` section 2, and they stay there). D-001 added. Evidence: the user's request for the Methods subsection of the co-culture paper, 2026-11-10. |
| 2026-11-10 (later) | D-002 added (the user's review of the drafted Methods: three SBI choices and the placeholder feature list kept, the detection-alignment note dropped); a "See also" line added to D-001. Evidence: the user's reply in the same chat, 2026-11-10. |
| 2026-11-10 (evening) | D-003 added: the Methods drop the "same detection" clause and keep the model's cell numbers. D-002's reading of "Drop this" and D-001's "See also" line marked [corrected]: the user dropped the clause from the section, not only the note. Tag `[SANDBOX]` added. Evidence: the user's request "lets keep these numbers and rewrite these methods", 2026-11-10. |
| 2026-11-13 | Moved into the repository as `docs/project/ANN_decisions_and_ideas_log.md` (branch `main`), copied unchanged from project knowledge (`claude/SBI_decisions_and_ideas_log.md`, which stays there as a frozen snapshot); from now on only this copy is written. Evidence: the project instructions, DOCUMENT UPKEEP v2 (2026-11-12). |

Tags: `[USER]` the user's decision or statement; `[REPO]` read in the repository; `[KB]` project knowledge or a file the user uploaded; `[SANDBOX]` run here; `[OWN]` own reasoning.

## Index

| ID | Date | Binds | Decision | Status |
|---|---|---|---|---|
| D-001 | 2026-11-10 | GiuliaGate SBI observation x_o; Methods of the co-culture paper (Main_V1) | The inference targets the control A50 (50:50 N:A) network | active |
| D-002 | 2026-11-10 | The SBI plan described in the Methods (D-001) | Amortised NPE with a conditional normalising flow; theta with 18 parameters; one x_o for the A50 group; feature list kept as a placeholder; "same detection" sentence kept unqualified **[corrected 2026-11-10: the clause is dropped, D-003]** | active, except the "same detection" item (superseded by D-003) |
| D-003 | 2026-11-10 | The Methods text; the model geometry it describes | "Same detection" clause dropped from the Methods; model cell numbers kept (108 + 108 on 300 um) | active (reading of "these numbers" to be confirmed) |

## Decisions

### D-001 (2026-11-10) -- SBI target: the control A50 cultures

- **Binds.** The observation x_o of the inference trained on the GiuliaGate simulations; the subsection "Computational model and simulation-based inference" of the co-culture paper (Main_V1.docx, placeholder "Modello Leo").
- **Decision** `[USER]`: use SBI "to find the parameters of our model that makes it reproduce the activity of the control network 50/50 N:A ratio".
- **Statement.** x_o is the summary-statistic vector of the spontaneous activity of the control A50 cultures (N:A 50:50, no Ara-C, recorded before any drug; n = 7 MEAs in the Electrophysiology table of Main_V1) `[KB]`. The target of inference is the posterior p(theta | x_o) over the GiuliaGate free parameters theta: the 16 free axes of `GiuliaGate/prior_config.tsv`, plus the per-network conn_prob and stoa_gate_p if they enter theta `[REPO]`.
- **Why.** The model network is 1:1 (Nn = Na = 108: 1,200 neurons and 1,200 astrocytes per mm^2 on a 300 um well, `GiuliaGate/launch_GG.sh`) `[REPO]`. The user states that cultures without any blocking drug reach 50:50 N:A whatever the plating ratio `[USER]` (2026-11-10), and asked that this not be argued in the Methods text.
- **Implies.** The Methods name the A50 controls as x_o and give the model's 1:1 ratio as matching A50, without arguing the convergence. For x to be "the same statistics" in simulation and experiment, the simulated readout must use the experimental detection settings (PTSD at 8 SD; log-ISI bursts of at least 10 spikes; network bursts with at least 30 % of channels bursting) `[KB]`; today the virtual-MEA detector thresholds on a robust (median-based) noise estimate and the MEA-style detector of `burst_metrics.py` uses at least 5 spikes and 80 % participation `[REPO]`, so the two must be aligned before inference `[OWN]`.
- **Does not decide.** The estimator (amortised NPE, as drafted in the Methods, or a sequential variant), the density estimator, the feature list, whether conn_prob and stoa_gate_p enter theta, per-culture versus pooled x_o, the analysis window, and whether other conditions (Ara-C, DHK, other ratios) become later targets.
- **Sources.** `[USER]` chat of 2026-11-10; `[KB]` Main_V1.docx (uploaded 2026-11-10); `[REPO]` main 62afa5b: `hpc/Phenomenological_finalv1/GiuliaGate/launch_GG.sh`, `GiuliaGate/prior_config.tsv`, `burst_metrics.py`; `MEA_Analysis_Reference.md` (virtual-MEA detection).
- **Status.** active.
- **See also** (added 2026-11-10). D-002 settles the estimator, the composition of theta and the single x_o that this entry left open, and records that the user dropped the alignment note above as a comment on the draft. **[corrected 2026-11-10]** The user dropped the "same detection" clause from the Methods text (D-003); the alignment requirement above is no longer stated in the paper, and whether the alignment work is done is still not decided.

### D-002 (2026-11-10) -- Methods: the SBI choices kept, the detection-alignment note dropped

- **Binds.** The subsection "Computational model and simulation-based inference" (D-001) and the SBI plan it describes; the defaults of the inference code when it is written.
- **Decision** `[USER]`, in reply to the notes on the draft: "Keep this" to the choices not yet in the code (amortised NPE with a normalising flow; conn_prob and stoa_gate_p counted in theta; one x_o for the whole A50 group); "Keep this" to the bracketed feature list as a placeholder; "Drop this" to the note that the simulated readout does not yet use the experimental detection settings. **[corrected 2026-11-10]** "Drop this" applied to the clause of the Methods that the note was about ("analysed with the same spike, burst and network-burst detection as the recordings"), not only to the note: it was misread as "keep the clause, drop the note" (D-003).
- **Statement.** The Methods describe neural posterior estimation with a conditional normalising flow q_phi(theta | x) (sbi toolbox), amortised, trained on N ~ 10^6 pairs (theta_n, x_n) with theta_n drawn from the prior; theta has 18 components: the 16 free axes of `GiuliaGate/prior_config.tsv` plus the per-network conn_prob and stoa_gate_p `[REPO]`; the trained flow is evaluated at one x_o for the control A50 group. The feature list stays in brackets until the Results fix it. The sentence "analysed with the same spike, burst and network-burst detection as the recordings" stays as written. **[corrected 2026-11-10]** That sentence is dropped (D-003); the rest of this statement stands.
- **Why.** The user's review of the drafted section, 2026-11-10 `[USER]`.
- **Implies.** When the inference code is written, these are its defaults: amortised (non-sequential) NPE; theta built from the 16 free columns plus conn_prob and stoa_gate_p read from the per-network records (they are drawn per network and stored beside the registry vector, not inside it) `[REPO]`; a single x_o for the A50 group. A sequential variant, a 16-D theta and per-MEA posteriors are opt-in comparisons, labelled as such `[OWN]`.
- **Does not decide.** The flow architecture and training settings; the final feature list; how x_o is pooled over the 7 MEAs (mean, median, or another summary); whether the alignment of the simulated detection named in D-001 "Implies" will be done (the user dropped it as a note on the draft; nothing was said about the work itself); the model's cell density (question open in the same chat, 2026-11-10). **[corrected 2026-11-10]** Cell density: settled by D-003.
- **Sources.** `[USER]` chat of 2026-11-10; `[REPO]` main 62afa5b: `GiuliaGate/prior_config.tsv`, `HPC_main_sweep.py` (topology-level axes recorded per network).
- **Status.** active, except the "same detection" sentence, superseded by D-003.

### D-003 (2026-11-10) -- Methods rewrite: "same detection" clause dropped, model cell numbers kept

- **Binds.** The subsection "Computational model and simulation-based inference" of the co-culture paper; the model geometry it describes (GiuliaGate as launched: Nn = Na = 108 on a 300 um x 300 um square, `GiuliaGate/launch_GG.sh`) `[REPO]`.
- **Decision** `[USER]`: "lets keep these numbers and rewrite these methos" (2026-11-10), after the density discussion; with the earlier "Drop this" (D-002).
- **Statement.** (a) The Methods no longer state that the simulated activity is analysed with the same spike, burst and network-burst detection as the recordings; they state that spikes, bursts and network bursts are detected on the virtual-electrode signals and that simulated and recorded activity are summarised by the same vector x of statistics. (b) Read as: the model keeps its current cell numbers, 108 neurons and 108 astrocytes on 300 um x 300 um (1,200 of each type per mm^2), rather than 54 + 54 on 300 um or 108 + 108 on a 424 um square (600 of each type per mm^2) `[OWN]` reading, to be confirmed by the user.
- **Why.** `[USER]`. Context from the same chat: at fixed cell numbers and conn_prob, halving the density by widening the square to 424 um leaves the neuron wiring and the SIC pairs identical (20/20 networks), removes 0.73 % of the gap junctions (150 um cap) and keeps the synapse-astrocyte link statistics within 2 %, while the neurons within 52 um of a virtual electrode drop from 10.16 to 4.95 per electrode (`check_density_scaling.py`, 20 rebuilt GiuliaGate networks, conn_prob ~ U[0.05, 0.4], gate 1) `[SANDBOX]`; halving the cell numbers instead halves the in-degree K = conn_prob (Nn - 1) at every conn_prob `[OWN]`.
- **Implies.** The Methods give the patch side and the cell numbers, not a density. The model's density (1,200 neurons and 1,200 astrocytes per mm^2) differs from the plating density stated in Main_V1 (1,200 cells per mm^2 in total) `[KB]`; the density of the cultures on the recording day (NeuN/GFAP counts) has not been compared with it.
- **Does not decide.** Whether the simulated detection will be aligned with the experimental settings (D-001, "Implies"); whether the virtual-MEA readout is ever re-run on positions scaled by sqrt(2) (the 424 um option; not adopted).
- **Sources.** `[USER]` chat of 2026-11-10; `[REPO]` main 62afa5b: `GiuliaGate/launch_GG.sh`, `HPC_single_run.py` (`build_topology`); `[KB]` Main_V1.docx, `MEA_Analysis_Reference.md` (52 um detection radius); `[SANDBOX]` `check_density_scaling.py`.
- **Status.** active; the reading of "these numbers" in (b) is to be confirmed, and an entry superseding (b) follows if the 424 um version was meant.

## Ideas

(none yet)
