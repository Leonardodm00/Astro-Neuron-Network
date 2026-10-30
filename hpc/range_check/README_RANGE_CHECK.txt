range_check -- 20 s runs over a setup's prior boxes, with the clamped model
===========================================================================
Question: with V held at VR during t_ref, what activity do a campaign's prior
boxes produce? Answered through the setup's own sweep code (its folder's
HPC_main_sweep.py), before the campaign is relaunched. Spec: specs/SPEC.md,
Block 6. Decided 2026-10-30: Giulia's setup and LindaColloDUP's now, one
48-core job each; MEAMA and the rho1300 family later.

THIS FOLDER (repository path hpc/range_check/)
  range_check.py            driver: one phase per call (oat or prior)
  range_analysis.py         per-run metrics (window after 2 s) and summaries
  range_plots.py            figures from the summaries
  submit_range_check.sh     PBS job: one setup, 48 cores, all phases + analysis
  smoke_test_range_check.py the checks (33 with Brian2; R8 tests the job script)
  README_RANGE_CHECK.txt    this file
range_analysis.py imports burst_metrics.py from ../Phenomenological_finalv1/
(so run it from a clone, or give --burst-lib DIR).

DESIGN (per setup)
  oat    every free axis at 7 values across its box (log-spaced on log axes),
         the others at nominal; 2 topologies at each conn_prob 0.1, 0.35, 0.6
  prior  8 topologies x 48 draws from the setup's own prior (conn_prob too)
  Each run is 20 s; metrics use the window 2-20 s.

BEFORE SUBMITTING (davinci)
  1. the setup folder runs the clamped model: python3 clamp_cadex_copies.py
     reports CLAMPED for it (range_check.py refuses an unclamped model)
  2. LindaColloDUP: its folder exists and is patched (APPLY_LCDUP.txt
     steps 0-5a: copy from MEAMA/, freeze patch, smoke test, clamp check)
  3. Giulia: the gate box of the new setup, if its HPC_main_sweep.py has
     --stoa_gate_lo/hi; without EXTRA the job uses the hhgap_v5 settings only

ON DAVINCI, from <clone>/hpc/range_check after git pull:
  conda activate brian_final && python smoke_test_range_check.py --no-sim | tail -1
      (expect "RESULT: 26 passed, 0 failed, 1 skipped"; without --no-sim it
       also simulates a tiny network, about 2 min: 33 passed)
  qsub -q cpu -v SETUP=lcdup submit_range_check.sh
  qsub -q cpu -v SETUP=giulia,EXTRA="--stoa_gate_lo 0.2 --stoa_gate_hi 1.0",OAT_EXTRA="--stoa_gate_lo 0.6 --stoa_gate_hi 0.6" submit_range_check.sh
  (the gate values above are an example: put the new setup's box in EXTRA and
   one fixed value inside it in OAT_EXTRA. qsub -v splits on commas: a freeze
   list other than LindaColloDUP's goes in as FREEZE_ENC, ':' for ',' and '@'
   for '=', as for submit_LCDUP.sh)

OUTPUT: ~/ANN/range_check/<setup>_<UTC>/ (outside every project folder)
  oat_cp0.100/ oat_cp0.350/ oat_cp0.600/ prior/   standard task layout
  runs.csv            one row per run: axis, value, conn_prob, metrics
  oat_summary.csv     per conn level, axis, value: median/min/max mean rate,
                      burst rate, active fraction
  prior_summary.txt   rate quantiles, shares silent / bursting / > 20 Hz,
                      Spearman ranking of the axes
  oat_curves.png, prior_rates.png, prior_spearman.png
Send back prior_summary.txt and the three png files.
Log: ~/range_check.o<jobid>. Runtime not measured on davinci (sandbox: one
20 s run of 159 neurons at nominal took 132 s on one core).
