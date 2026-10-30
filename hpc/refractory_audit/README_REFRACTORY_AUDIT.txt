Refractory-runaway audit for existing campaigns (MEAMA, Giulia's simulations)
==============================================================================
Question: in a saved campaign, how many runs lost neurons to the refractory
runaway (V -> NaN during t_ref, the neuron never fires again)?
Background, rules and calibration: project doc claude/Refractory_NaN_Audit.md

THIS FOLDER (repo path hpc/refractory_audit/)
  check_refractory_nan.sh         the audit (login node): CODE, LOGS, DATA, VERDICT
  refractory_scan.py              the spike-train screen it calls (numpy only)
  submit_refractory_scan.sh       the same screen on EVERY run, as a PBS job
  smoke_test_refractory_scan.py   checks the two scripts on synthetic campaigns
  README_REFRACTORY_AUDIT.txt     this file
  .gitignore                      keeps the output folders out of git
Nothing here writes into a campaign: results go to refractory_audit_<UTC>/
next to the scripts.

ORDER OF OPERATIONS ON DAVINCI (paste one line at a time)

0. bring the folder in with git. Find your clone of the repository (prints
   REPO=none if there is none on davinci):
REPO=$(grep -il astro-neuron-network ~/*/.git/config ~/*/*/.git/config ~/*/*/*/.git/config ~/*/*/*/*/.git/config 2>/dev/null | head -1 | sed 's|/\.git/config$||'); echo "REPO=${REPO:-none}"
   only if REPO=none, clone it (about 2 MB):
cd ~/ANN && git clone https://github.com/Leonardodm00/astro-neuron-network.git Astro-Neuron-Network && REPO=~/ANN/Astro-Neuron-Network && echo "REPO=$REPO"
   then pull (first line must read "## main...origin/main"):
cd "$REPO" && git status -sb | head -6 && git pull && git log --oneline -1 && ls -A hpc/refractory_audit

1. bytes (expect "0 CR 0 non-ASCII" four times):
cd "$REPO/hpc/refractory_audit" && python3 -c "import sys;[print(p, open(p,'rb').read().count(b'\r'), 'CR', sum(b>127 for b in open(p,'rb').read()), 'non-ASCII') for p in sys.argv[1:]]" check_refractory_nan.sh refractory_scan.py submit_refractory_scan.sh smoke_test_refractory_scan.py

2. environment and smoke test (expect "RESULT: 41 passed, 0 failed"):
conda activate brian_final && python smoke_test_refractory_scan.py | tail -1

3. the audit on MEAMA n01 and Giulia's v1 v2 v4 v5, 300 random runs each
   (a few minutes; drop --sample 300 for 1,000 each):
bash check_refractory_nan.sh --sample 300

   Other campaigns:  bash check_refractory_nan.sh --sample 300 /path/campaign_X /path/campaign_Y

4. only if you want the per-run tables of EVERY run (to filter a training set):
qsub -q cpu submit_refractory_scan.sh

Without git: copy refractory_audit_pkg.tar.gz into an empty folder with the
MobaXterm file browser (binary transfer), tar -xzf it there, then steps 1-4.

WHAT THE OUTPUT SAYS
  1. CODE    NOT CLAMPED (exposed) = the campaign's ASD_fun_BD_cpp.py has no
             "(unless refractory)" on its dV/dt line; CLAMPED = not exposed.
             A WARNING line means the file changed after the campaign started.
  2. LOGS    per job name: PBS logs, and how many carry Brian2's
             "Neuron's variable 'V' has NaN" (once per worker process at most,
             so a lower bound). meama_campaign = MEAMA, asn_campaign = the
             rho1300 / Giulia worker. Old logs may have been deleted.
  3. DATA    per campaign: the share of runs with >= 1 neuron whose last spike
             came in the first second and that stayed silent to the end (the
             calibrated rule), with a 95 % interval when sampled; the share
             of runs where the whole network went silent that way; the share
             of neurons; incidence by tercile of each swept axis.
  VERDICT    one block per campaign. Send it back in the chat.
