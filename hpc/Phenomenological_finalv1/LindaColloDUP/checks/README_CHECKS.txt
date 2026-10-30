Standalone checks behind the LindaColloDUP settings
===================================================
Not pipeline code. Sandbox checks of 2026-10-29/30 (Brian2 2.11.0, copies of
the 36-D main pipeline). Results are recorded in the project doc
claude/LindaColloDUP_Campaign_Operations.md (LCO). Pure ASCII.

check_rs_reset_adaptation.py
    Question: do the frozen V_A, Delta_A, V_R give the adapting (regular
    spiking) CAdEx cell of Gorski et al. 2021 (Table 1, "Adaptive spiking"),
    and which (V_A, V_R) keeps the reset below the V-nullcline minimum?
    Run:    python3 -I check_rs_reset_adaptation.py
    Result: LCO section 2.1.

check_neuronal_inert_axes.py
    Claim: under MODE=Neuronal, alpha_syn, O_G and Omega_G leave the spike
    output bit-identical at fixed seeds, while a live axis (U_0_sr) changes it.
    Run:    python3 -I check_neuronal_inert_axes.py --pipeline_dir D --work_dir W
            (D holds HPC_single_run.py, ASD_fun_BD_cpp.py, synapse_pdist.csv)
    Result: LCO section 2 (the three axes are frozen and not counted).

check_sigma_meaning.py
    Claim: with the code's noise term sigma*sqrt(2*gl_ref/Cm)*randn()/sqrt(dt),
    the stationary SD of V of the passive membrane equals sigma (tolerance 2 %).
    Run:    python3 check_sigma_meaning.py
    Result: SD(V) = 3.997 mV for sigma = 4 mV (ratio 0.9991), PASS (2026-10-30).

print_job_args.py
    Prints the job_args.json the LindaColloDUP settings produce (free axes,
    freeze list, Nn, Na, mode, simtime, the boxes) without simulating.
    Run:    python3 print_job_args.py <folder with the freeze-patched pipeline>
