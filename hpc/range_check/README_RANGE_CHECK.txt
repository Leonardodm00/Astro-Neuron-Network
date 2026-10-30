range_check -- 20 s runs over a setup's prior boxes, with the clamped model
===========================================================================
Question: with V held at VR during t_ref, what activity do a campaign's prior
boxes produce? Answered through the setup's own sweep code (its folder's
HPC_main_sweep.py), before the campaign is relaunched. Spec: specs/SPEC.md,
Block 6. Decided 2026-10-30: Giulia's setup and LindaColloDUP's now, each on
two 48-core nodes (launch_range_check.sh); MEAMA and the rho1300 family later.

THIS FOLDER (repository path hpc/range_check/)
  range_check.py            driver: one phase per call (oat or prior)
  range_analysis.py         per-run metrics (window after 2 s) and summaries
  range_plots.py            figures from the summaries
  launch_range_check.sh     login node: one setup as three PBS jobs (below)
  submit_range_check.sh     the PBS job; PHASES and ANALYZE choose its part
                            (submitted alone, it runs everything on one node)
  smoke_test_range_check.py the checks (50 with Brian2; R8 tests the job
                            script and R9 the launcher, both with stubs)
  README_RANGE_CHECK.txt    this file
range_analysis.py imports burst_metrics.py from ../Phenomenological_finalv1/
(so run it from a clone, or give --burst-lib DIR).

DESIGN (per setup)
  oat    every free axis at 7 values across its box (log-spaced on log axes),
         the others at nominal; 2 topologies at each of three conn_prob
         levels, the ends and the middle of the setup's conn_prob box:
           lcdup   0.01  0.255  0.5        giulia  0.1  0.35  0.6
  prior  16 topologies x 48 draws from the setup's own prior (conn_prob too)
  Each run is 20 s; metrics use the window 2-20 s.
  LindaColloDUP: 13 free axes x 7 values = 91 points, padded with 5 nominal
  runs to 96 = 2 rounds of 48 per topology; oat = 3 levels x 2 topologies x
  96 = 576 runs (12 rounds), prior = 16 x 48 = 768 runs (16 rounds).

TWO NODES (launch_range_check.sh, decided 2026-10-30)
  job 1  rc_oat       queue intel  48 cores  PHASES=oat    the oat phase
  job 2  rc_prior     queue cpu    48 cores  PHASES=prior  the prior sample
  job 3  rc_analysis  queue cpu     8 cores  analysis and figures, held until
         jobs 1 and 2 have ended (depend=afterany: it also runs if one of them
         failed; the summaries then show which phases are there)
  All three write into one OUT. Before submitting, the launcher runs the job
  script's own checks once on the login node (CHECK_ONLY=1): a setup folder
  that is not ready stops it, with nothing submitted. Variables: Q_OAT,
  Q_PRIOR, Q_ANALYSIS, NCPUS (48), WALLTIME (12:00:00), PRIOR_TOPOS (16),
  ANALYSIS_NCPUS (8), ANALYSIS_WALLTIME (02:00:00), OUT, DRYRUN.

BEFORE SUBMITTING (davinci)
  1. the setup folder runs the clamped model: python3 clamp_cadex_copies.py
     reports CLAMPED for it (range_check.py refuses an unclamped model)
  2. LindaColloDUP: its folder in the clone,
     ~/ANN/Astro-Neuron-Network/hpc/Phenomenological_finalv1/LindaColloDUP,
     set up with APPLY_LCDUP.txt steps 0-5a (pipeline copied from MEAMA/,
     freeze and box patches, smoke test, clamp). Without the two g box
     markers the job script stops (exit 2), and so does the launch.
  3. Giulia: the gate box of the new setup, if its HPC_main_sweep.py has
     --stoa_gate_lo/hi; without EXTRA the job uses the hhgap_v5 settings only

ON DAVINCI, from ~/ANN/Astro-Neuron-Network/hpc/range_check after git pull:
  conda activate brian_final && python smoke_test_range_check.py --no-sim | tail -1
      (expect "RESULT: 43 passed, 0 failed, 1 skipped"; without --no-sim it
       also simulates a tiny network, about 1-2 min: 50 passed)
  DRYRUN=1 SETUP=lcdup bash launch_range_check.sh     (prints the 3 qsub lines)
  SETUP=lcdup bash launch_range_check.sh              (submits them)
  SETUP=giulia EXTRA="--stoa_gate_lo 0.2 --stoa_gate_hi 1.0" OAT_EXTRA="--stoa_gate_lo 0.6 --stoa_gate_hi 0.6" bash launch_range_check.sh
  (the gate values above are an example: put the new setup's box in EXTRA and
   one fixed value inside it in OAT_EXTRA. No commas in any variable: qsub -v
   splits on them, so a freeze list other than LindaColloDUP's goes in as
   FREEZE_ENC, ':' for ',' and '@' for '=', as for submit_LCDUP.sh)
  One node only: qsub -q cpu -v SETUP=lcdup submit_range_check.sh
  (all phases and the analysis in one job; 8 prior topologies unless
   PRIOR_TOPOS is set)

OUTPUT: ~/ANN/range_check/<setup>_<UTC>/ (outside every project folder)
  oat_cp<level>/ per level (lcdup: oat_cp0.010/ oat_cp0.255/ oat_cp0.500/)
  and prior/, in the standard task layout
  runs.csv            one row per run: axis, value, conn_prob, metrics
  oat_summary.csv     per conn level, axis, value: median/min/max mean rate,
                      burst rate, active fraction
  prior_summary.txt   rate quantiles, shares silent / bursting / > 20 Hz,
                      Spearman ranking of the axes
  oat_curves.png, prior_rates.png, prior_spearman.png
Send back prior_summary.txt and the three png files.
Logs (#PBS -k eo, in $HOME): rc_oat.o<id>, rc_prior.o<id>, rc_analysis.o<id>
and their .e<id>; a job submitted alone logs to range_check.o<id>.
Runtime not measured on davinci (sandbox: one 20 s run of 159 neurons at
nominal took 132 s on one core); the prior job, 16 rounds, is the longer.
