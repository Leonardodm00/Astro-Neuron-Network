Poster / publication

## Bursts

### Run the smoke tests (each builds its own synthetic data, asserts correctness, exits non-zero on failure):
python burst_palettes.py          # palette legibility + A0 sizing checks
python burst_poster_plots.py      # renders every figtype x palette + degenerate case
python run_burst_analysis.py --smoke-test   # full pipeline on a synthetic campaign
### Run for real — on the cluster, or directly:
qsub -v CAMPAIGN=campaign_<TAG> run_burst_analysis.sh
bash run_burst_analysis.sh campaign_<TAG>          # login-node / quick look
