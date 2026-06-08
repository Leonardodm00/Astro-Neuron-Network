Main scripts for simulation






# Sweep status

python status_sweep.py campaign_300k_v1
python status_sweep.py campaign_300k_v1 --sample 10000   # more precise stats
python status_sweep.py campaign_300k_v1 --json           # for logging / scripting
watch -n 300 python status_sweep.py campaign_300k_v1    # auto-refresh every 5 minStatus sweepPY Open in Visual Studio Code
