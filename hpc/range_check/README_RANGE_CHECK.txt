range_check -- the standard prior range calibration, with the clamped model
===========================================================================
Question: does a campaign's prior box produce plausible activity across its
whole range? Answered through the setup's own sweep code (its folder's
HPC_main_sweep.py) before the campaign is (re)launched. Spec: specs/SPEC.md,
Block 6. Decided 2026-10-30 as the standard for every setup: the corners of
the box plus random cells of its 7-bin grid, on two 48-core nodes. Giulia's
setup and LindaColloDUP's now; MEAMA and the rho1300 family later.

THIS FOLDER (repository path hpc/range_check/)
  range_design.py           the designs (pure functions, numpy only)
  range_check.py            driver: one level of one part per call
  range_analysis.py         per-run metrics, labels and summaries
  range_plots.py            figures from the summaries
  launch_range_check.sh     login node: one setup as three PBS jobs (below)
  submit_range_check.sh     the PBS job; PHASES and ANALYZE choose its part
                            (submitted alone, it runs everything on one node)
  smoke_test_range_check.py the checks (63 with Brian2; R9 tests the job
                            script and R10 the launcher, both with stubs)
  README_RANGE_CHECK.txt    this file
range_analysis.py imports burst_metrics.py from ../Phenomenological_finalv1/
(so run it from a clone, or give --burst-lib DIR).

DESIGN (per setup). Dimensions: the free axes (sweep group minus freeze) in
the sweep's theta coordinate (ln on log-uniform axes) plus conn_prob.
  corners   every combination of each dimension's lower and upper bound: the
            full two-level factorial when 2^d <= 16,384, otherwise a
            two-level fraction of 16,384 runs whose resolution is >= V (no
            main effect or two-axis interaction aliased with another).
            LindaColloDUP: 13 axes + conn_prob = 14, so the full factorial,
            8,192 corners at conn_prob 0.01 and 8,192 at 0.5, each half on 8
            topologies with 256 centre points (the box centre) filling the
            last slots: 16,896 runs.
  interior  each dimension cut into 7 bins of equal width in its own
            coordinate (equal prior mass); a cell takes the bin midpoints,
            the bin drawn uniformly and independently per dimension.
            LindaColloDUP: conn_prob at the 7 bin midpoints 0.045 ... 0.465,
            2,352 cells each (7 topologies x 48 workers x 7): 16,464 runs.
  Each run is 20 s; metrics use the window 2-20 s.

TWO NODES (launch_range_check.sh, decided 2026-10-30)
  job 1  rc_corners   queue intel  48 cores  24 h  PHASES=corners
  job 2  rc_interior  queue cpu    48 cores  24 h  PHASES=interior
  job 3  rc_analysis  queue cpu     8 cores   2 h  analysis and figures, held
         until jobs 1 and 2 have ended (depend=afterany: it also runs if one
         of them failed; the summaries then show which parts are there)
  All three write into one OUT. Before submitting, the launcher runs the job
  script's own checks once on the login node (CHECK_ONLY=1): a setup folder
  that is not ready stops it, with nothing submitted. Variables: Q_CORNERS,
  Q_INTERIOR, Q_ANALYSIS, NCPUS (48), WALLTIME (24:00:00), ANALYSIS_NCPUS (8),
  ANALYSIS_WALLTIME (02:00:00), OUT, DRYRUN; passed on to the jobs: BINS (7),
  CORNER_LEVELS (0 1), CORNER_TOPOS (8), CORNER_BUDGET (16384),
  INTERIOR_LEVELS (0..BINS-1), INTERIOR_TOPOS (7), INTERIOR_RUNS (16384),
  RC_SIMTIME (20), LIB_DIR, LIKE, EXTRA, FREEZE_ENC, CONDA_ENV.

BEFORE SUBMITTING (davinci)
  1. the setup folder runs the clamped model: python3 clamp_cadex_copies.py
     reports CLAMPED for it (range_check.py refuses an unclamped model)
  2. LindaColloDUP: its folder in the clone,
     ~/ANN/Astro-Neuron-Network/hpc/Phenomenological_finalv1/LindaColloDUP,
     set up with APPLY_LCDUP.txt steps 0-5a (pipeline copied from MEAMA/,
     freeze and box patches, smoke test, clamp). Without the two g box
     markers the job script stops (exit 2), and so does the launch.
  3. Giulia: the gate box of the new setup, if its HPC_main_sweep.py has
     --stoa_gate_lo/hi, goes in EXTRA; the gate is then drawn per topology
     from its prior, not designed (it is not a design dimension yet)

ON DAVINCI, from ~/ANN/Astro-Neuron-Network/hpc/range_check after git pull:
  conda activate brian_final && python smoke_test_range_check.py --no-sim | tail -1
      (expect "RESULT: 56 passed, 0 failed, 1 skipped"; without --no-sim it
       also simulates a tiny network, about 1 min: 63 passed)
  DRYRUN=1 SETUP=lcdup bash launch_range_check.sh     (prints the 3 qsub lines)
  SETUP=lcdup bash launch_range_check.sh              (submits them)
  SETUP=giulia EXTRA="--stoa_gate_lo 0.2 --stoa_gate_hi 1.0" bash launch_range_check.sh
  (the gate values above are an example. No commas in any variable: qsub -v
   splits on them, so a freeze list other than LindaColloDUP's goes in as
   FREEZE_ENC, ':' for ',' and '@' for '=', as for submit_LCDUP.sh)
  One node only: qsub -q cpu -v SETUP=lcdup submit_range_check.sh
  (both parts and the analysis in one job, about twice as long)
  To see a level's design without running it:
  python range_check.py --rc-phase corners --rc-level 0 --rc-out /tmp/x --rc-design-only <settings>

OUTPUT: ~/ANN/range_check/<setup>_<UTC>/ (outside every project folder)
  corners_c0/ corners_c1/ interior_b0/ ... interior_b6/, in the standard task
  layout, each with its design.json
  runs.csv                one row per run: kind (corner, centre, interior),
                          codes or bins, theta and natural values, metrics
  corner_effects.csv      per dimension: upper minus lower bound, averaged
                          over all other corners, on log10 rate and on the
                          shares silent / > 20 Hz / bursting
  corner_interactions.csv every two-dimension interaction on log10 rate
  interior_bins.csv       per dimension and bin: rate quartiles and shares
  interior_pairs.csv      per pair of dimensions and pair of bins
  interior_spearman.csv   rank correlation of each dimension with the rate
  calibration_summary.txt all of it in short, with the centre-point curvature
  corner_effects.png, interior_bins.png, rate_distributions.png
Send back calibration_summary.txt and the three png files.
Logs (#PBS -k eo, in $HOME): rc_corners.o<id>, rc_interior.o<id>,
rc_analysis.o<id> and their .e<id>; a job submitted alone logs to
range_check.o<id>. Runtime not measured on davinci (sandbox: one 20 s run of
159 neurons at nominal took 132 s on one core; runs at conn_prob 0.5 are
slower than at 0.01): each node runs about 350 rounds of 48 runs.
