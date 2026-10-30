Standalone checks behind the numbers of the refractory audit
============================================================
Not pipeline code. Each script tests one claim on simulated data; all were
run in the conversation's sandbox (Brian2 2.11.0, numpy, scipy; copies of
the 36-D main pipeline) on 2026-10-29/30. Results are recorded in the
project docs claude/Refractory_NaN_Audit.md (RNA) and
claude/LindaColloDUP_Campaign_Operations.md (LCO). Pure ASCII; no paths are
hard-coded, every input is a command-line argument.

check_blowup_time.py
    Claim: without a threshold (as during t_ref when dV/dt lacks
    "(unless refractory)") the CAdEx voltage equation, started above its
    unstable fixed point, reaches +inf in finite time. Registry cell, no
    adaptation, no input, no noise.
    Run:    python3 check_blowup_time.py
    Result: RNA section 1 (V_u = -47.68 mV; 0.92 ms from VT + 3 DeltaT).

check_reset_runaway_input.py
    Claim: a constant net depolarizing current lowers the unstable fixed
    point and removes it above the rheobase; smallest constant current that
    takes V from V_R to +inf within t_ref = 3 ms (gA = 0, no noise).
    Run:    python3 check_reset_runaway_input.py
    Result: rheobase 5.98 pA; 70.0 pA from V_R = -55 mV, 90.3 pA from -58.2 mV.

check_refractory_blowup.py
    Claim: in the production worker (HPC_main_sweep._worker_entry, run
    in-process, Brian2's post-run state check intercepted) neurons end runs
    with V NaN or |V| > 1e50 unless V is held during t_ref.
    Run:    python3 check_refractory_blowup.py run --pipeline_dir D --work_dir W \
                --label L [--clamp | --allow_clamped_source] \
                --n_topo 6 --k 4 --simtime 5 --seed 2026
            python3 check_refractory_blowup.py compare W/result_a.json W/result_b.json
    D holds HPC_main_sweep.py (with the LindaColloDUP --freeze patch),
    HPC_single_run.py, ASD_fun_BD_cpp.py and synapse_pdist.csv. --clamp adds
    the flag to the private copy; --allow_clamped_source runs a file that
    already has it (the patched repository file) as it is.
    Result: LCO section 3 (23 of 24 runs with NaN neurons as shipped, 0 of 24
    clamped; patched repository file: 0 NaN in the paired runs).

calibrate_refractory_scan.py
    Claim: the rules of ../refractory_scan.py flag the neurons that really
    went NaN and no healthy ones.
    Run:    python3 calibrate_refractory_scan.py --scan_dir .. --results W/result_*.json
    Result: RNA section 4.

check_nan_last_spike_timing.py
    Claim: most NaN neurons fired their last spike at the start of the run.
    Run:    python3 check_nan_last_spike_timing.py W/result_*.json
    Result: RNA section 1 (72-76 % within 10 ms, 85-90 % within 100 ms).
