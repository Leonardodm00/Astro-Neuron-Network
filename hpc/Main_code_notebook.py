# =============================================================================
# Main_code_notebook.py — HPC / Colab entry point
#
# Builds the neuron+astrocyte network once, compiles to cpp_standalone, then
# re-runs the compiled binary many times with different parameter vectors via
# `device.run(run_args=..., seed=...)`. Each run saves a small .npz with the
# spike times of neurons and astrocytes plus the parameter vector used.
#
# Revision applies the fixes documented in Report 3:
#   A.1 astrocyte warmup ICs (handled inside Astrocyte_Group)
#   A.7 inconsistency awareness vs Main_code_PRINCIPAL
#   B.3 conn_prob_ defined in the 'Neuronal' branch
#   C.5 drop StateMonitor; save per-iteration spike .npz
#   C.6 expose seed via device.run(seed=...)
#   C.7 Binomial_fun passed as argument to Neuronal_Network
#   D.1 set_device() before device.reinit()/device.activate()
#   D.2 verbose flag gates per-iteration prints
#   D.6 import torch removed
# =============================================================================

# --- Environment setup (Colab) -----------------------------------------------
!pip install brian2 -q
!apt-get update
!apt-get install libgsl-dev -y
!pip install scikit-optimize

from brian2 import *
import numpy as np
from time import time
import os
import pandas as pd

from skopt.space import Real, Integer

from google.colab import drive
drive.mount('/content/drive', force_remount=True)

# Change the current working directory to where ASD_fun_BD_cpp.py lives.
# The module name on the user's drive is `ASN_fun_BD_cpp`; rename here if
# different on your side.
try:
    os.chdir('/content/drive/MyDrive')
    from ASN_fun_BD_cpp import *
except FileNotFoundError:
    print("Error: directory not found in Google Drive. Check the path.")
except ModuleNotFoundError:
    print("Error: ASN_fun_BD_cpp.py not found in the specified directory.")


# --- Brian2 logging suppression ----------------------------------------------
BrianLogger.suppress_hierarchy('brian2.devices')
BrianLogger.suppress_hierarchy('brian2.parsing')

start_scope()


# --- SET DEVICE (must come BEFORE device.reinit/activate; Report 3 §D.1) -----
set_device('cpp_standalone', build_on_run=False)
device.reinit()
device.activate()


# =============================================================================
# Custom Brian2 Function: Binomial_fun
# =============================================================================
def _Binomial_fun_py(n, p, _vectorisation_idx):
    """Pure-Python fallback (Cython/cpp implementations follow)."""
    uniform = np.random.rand(n)
    return sum(uniform < p)


Binomial_fun = Function(_Binomial_fun_py,
                        arg_units=[1, 1], return_unit=1,
                        stateless=False, auto_vectorise=True)

cython_code = '''
cdef double Binomial_fun(int n, double p, _vectorisation_idx):
    cdef int count = 0
    cdef double uniform
    cdef int i
    for i in range(n):
        uniform = rand(_vectorisation_idx)
        if uniform < p:
            count = count + 1
    return count;
'''

cpp_code = '''
int Binomial_fun(int n, double p, int _vectorisation_idx) {
    int count = 0;
    for (int i = 0; i < n; ++i) {
        if (rand(_vectorisation_idx) < p) {
            count = count + 1;
        }
    }
    return count;
}
'''

Binomial_fun.implementations.add_implementation(
    'cython', cython_code,
    dependencies={'rand': DEFAULT_FUNCTIONS['rand']},
)
Binomial_fun.implementations.add_implementation(
    'cpp', cpp_code,
    dependencies={'rand': DEFAULT_FUNCTIONS['rand']},
)


# =============================================================================
# SimWrapper
# =============================================================================
class SimWrapper:
    """
    One-shot build of the full neuron+astrocyte network, then many-shot
    parameter-swept runs of the compiled C++ binary.

    Parameters
    ----------
    Binomial_fun : brian2.Function
        Required by the synapse equations.
    connections_path : str
        Folder containing the pre-built connectivity (S_/GJ_/StoA_*.npy and
        Neuron_/Astrocyte_state_namespace.npz). Produced locally by
        `save_synaptic_connections` in PRINCIPAL.
    out_dir : str
        Directory where per-iteration spike .npz files will be saved.
    simtime : Brian2 second
        Total simulated duration per run.
    Nn, Na : int
        Number of neurons and astrocytes. Must match the connectivity files.
    Simulated_network : {'Full', 'Neuronal', 'Astrocytic'}
    seed_device, seed_neuron, seed_synapse, seed_astro : int
        Seeds used at network-build time.
    verbose : bool
        If True, print per-iteration progress (Report 3 §D.2).

    Notes
    -----
    A.7: parameters here (notably `simtime`, the synapse coordinates, and the
    fact that connections are LOADED rather than regenerated) must match the
    PRINCIPAL run that produced the connectivity files. Worst case: a topology
    file labelled `conn_prob=0.13` is loaded by an HPC run whose comments say
    `conn_prob=0.107`. The `Neuron_state_namespace.npz` carries the actual
    value if you ever need to recover it.
    """

    def __init__(self, Binomial_fun, connections_path, out_dir,
                 simtime=30 * second,
                 Nn=100, Na=43,
                 Simulated_network='Full',
                 seed_device=50, seed_neuron=39,
                 seed_synapse=35, seed_astro=60,
                 verbose=False):

        self.out_dir = out_dir
        self.verbose = verbose
        os.makedirs(out_dir, exist_ok=True)

        devices.device.seed(seed_device)
        defaultclock.dt = 0.05 * ms

        # ------ Load connectivity from disk -----------------------------
        Connections_dict = extract_synaptic_connections(connections_path)
        Syn_pdist = None                                                 # not needed when connections are loaded

        # Synapse bouton positions (optional in old folders — backward compat).
        syn_positions = None
        if 'S_x_syn' in Connections_dict and 'S_y_syn' in Connections_dict:
            syn_positions = np.column_stack(
                [Connections_dict['S_x_syn'], Connections_dict['S_y_syn']]
            )

        # ------ Build groups --------------------------------------------
        if Simulated_network == 'Full':
            N, S = Neuronal_Network(
                Nn,
                Syn_pdist=Syn_pdist,
                ics=False,
                Simulated_network=Simulated_network,
                Decay_type='Double_exp',
                synapse_type='facilitating',
                conn_prob_=None,
                seed_neu=seed_neuron,
                seed_syn=seed_synapse,
                connections=[
                    Connections_dict['S_source'],
                    Connections_dict['S_target'],
                ],
                Binomial_fun=Binomial_fun,
                syn_positions=syn_positions,
            )
            N.I = '(rand() - 0.5) * I_inj'

            Astro, GJ = Astrocyte_Group(
                Na, Simulated_network,
                seed_astro=seed_astro,
                ics='steady',                                            # Report 3 §A.1
                connections=[
                    Connections_dict['GJ_source'],
                    Connections_dict['GJ_target'],
                ],
            )

            GT = Gliotransmission(Na, Astro, ics='jitter', seed_astro=seed_astro)

            StoA, Connections_list = Synapse_to_astro(
                S, Astro,
                connections=[
                    Connections_dict['StoA_source'],
                    Connections_dict['StoA_target'],
                ],
            )
            AtoS = Astro_to_Syn(GT, S, connections=Connections_list)

        elif Simulated_network == 'Neuronal':
            conn_prob_ = None                                            # Report 3 §B.3
            N, S = Neuronal_Network(
                Nn,
                Syn_pdist=Syn_pdist,
                ics=False,
                Simulated_network=Simulated_network,
                Decay_type='Double_exp',
                synapse_type='facilitating',
                conn_prob_=conn_prob_,
                seed_neu=seed_neuron,
                seed_syn=seed_synapse,
                connections=[
                    Connections_dict['S_source'],
                    Connections_dict['S_target'],
                ],
                Binomial_fun=Binomial_fun,
                syn_positions=syn_positions,
            )
            N.I = '(rand() - 0.5) * I_inj'

        elif Simulated_network == 'Astrocytic':
            Astro, GJ, P, Glu_Input = Astrocyte_Group(
                Na, Simulated_network,
                seed_astro=seed_astro,
                ics='steady',
            )

        # ------ Monitors (spikes only, Report 3 §C.5) -------------------
        SpikesN = SpikeMonitor(N, name='Spike_monitor_N')
        if Simulated_network in ('Full', 'Astrocytic'):
            SpikesA = SpikeMonitor(Astro, name='Spike_monitor_A')

        # ------ Build and compile the standalone binary -----------------
        self.net = Network()
        if Simulated_network == 'Full':
            self.net.add([N, S, Astro, GJ, GT, StoA, AtoS, SpikesN, SpikesA])
        elif Simulated_network == 'Neuronal':
            self.net.add([N, S, SpikesN])
        elif Simulated_network == 'Astrocytic':
            self.net.add([Astro, GJ, P, Glu_Input, SpikesA])

        self.net.run(simtime)
        self.device = get_device()
        self.device.build(run=False, directory=None)

        # Remember the requested mode so do_run knows which monitors to harvest.
        self._mode = Simulated_network

    # ------------------------------------------------------------------
    def do_run(self, params, iteration, seed_run=None):
        """
        Execute the compiled binary once with a new parameter vector.

        Parameters
        ----------
        params : array-like of length 14
            Order matches `space` defined at the bottom of this script
            (Sigma, g_AHP, Xi_ampa, Xi_nmda, Tau_Ca, U_0_ar, U_max,
             U_0_sr, Omega_f_sr, Omega_f_ar, Omega_d, alpha_syn,
             g_na, g_kd).
        iteration : int
            Used as the suffix in the per-iteration output filename
            (`iter_{iteration:07d}.npz`).
        seed_run : int or None
            If None, the binary replays its build-time RNG sequence (identical
            noise / initial randomisation across all calls — this is the
            paired-comparison regime). If integer, that seed is forwarded to
            `device.run(seed=...)` and the C++ RNG is re-initialised, so each
            iteration sees its own noise and `(rand()-0.5)*I_inj` realisation
            (Report 3 §C.6).

        Returns
        -------
        str : path to the saved .npz.
        """
        # Workaround so the device is set correctly inside this method.
        from brian2.devices import device_module
        device_module.active_device = self.device

        # --- Unpack the 14-D parameter vector ---
        # Neurons
        Sigma_     = params[0] * mV
        g_AHP_     = params[1] * nS
        # Synapse
        Xi_ampa_   = params[2] / mmole
        Xi_nmda_   = params[3] / mmole
        Tau_Ca_    = params[4] * second
        U_0_ar_    = params[5]
        U_max_     = params[6] / ms
        U_0_sr_    = params[7]
        Omega_f_sr_ = params[8] / second
        Omega_f_ar_ = params[9] / second
        Omega_d_    = params[10] / second
        # Syn-Neuron
        alpha_syn_ = params[11]
        # Ion channels
        area = self.net['Neuron'].namespace['area']
        g_na_ = params[12] * msiemens * cm**-2 * area
        g_kd_ = params[13] * msiemens * cm**-2 * area

        run_args = {
            # Synaptic
            self.net['Synapse'].U_0_ar:     U_0_ar_,
            self.net['Synapse'].Umax:       U_max_,
            self.net['Synapse'].U_0_sr:     U_0_sr_,
            self.net['Synapse'].Omega_f_sr: Omega_f_sr_,
            self.net['Synapse'].Omega_f_ar: Omega_f_ar_,
            self.net['Synapse'].Omega_d:    Omega_d_,
            self.net['Synapse'].alpha_syn:  alpha_syn_,
            self.net['Synapse'].Xi_ampa:    Xi_ampa_,
            self.net['Synapse'].Xi_nmda:    Xi_nmda_,
            # Neuron
            self.net['Neuron'].sigma:       Sigma_,
            self.net['Neuron'].g_AHP:       g_AHP_,
            self.net['Neuron'].tau_Ca:      Tau_Ca_,
            # Ion channels
            self.net['Neuron'].g_na:        g_na_,
            self.net['Neuron'].g_kd:        g_kd_,
        }

        if seed_run is None:
            self.device.run(run_args=run_args)
        else:
            self.device.run(run_args=run_args, seed=seed_run)

        # --- Harvest spike data and save to disk ---
        spk_N_t = np.asarray(self.net['Spike_monitor_N'].t / second)
        spk_N_i = np.asarray(self.net['Spike_monitor_N'].i)

        if self._mode in ('Full', 'Astrocytic'):
            spk_A_t = np.asarray(self.net['Spike_monitor_A'].t / second)
            spk_A_i = np.asarray(self.net['Spike_monitor_A'].i)
        else:
            spk_A_t = np.array([], dtype=float)
            spk_A_i = np.array([], dtype=int)

        out_path = os.path.join(self.out_dir, f'iter_{iteration:07d}.npz')
        np.savez_compressed(
            out_path,
            params=np.asarray(params, dtype=float),
            spk_N_t=spk_N_t.astype(np.float32),
            spk_N_i=spk_N_i.astype(np.int32),
            spk_A_t=spk_A_t.astype(np.float32),
            spk_A_i=spk_A_i.astype(np.int32),
            seed_run=np.int64(-1 if seed_run is None else seed_run),
        )

        if self.verbose:
            print(f'  iter {iteration:07d}: {len(spk_N_t):6d} N-spikes, '
                  f'{len(spk_A_t):4d} A-events -> {out_path}')

        return out_path


# =============================================================================
# Main entry point
# =============================================================================
if __name__ == "__main__":

    CONNECTIONS_PATH = r'/content/drive/MyDrive/Output connectivity'
    OUT_DIR          = r'/content/drive/MyDrive/HPC_sweep_results'

    sim = SimWrapper(
        Binomial_fun,
        connections_path=CONNECTIONS_PATH,
        out_dir=OUT_DIR,
        simtime=30 * second,
        Nn=100, Na=43,
        Simulated_network='Full',
        seed_device=50, seed_neuron=39,
        seed_synapse=35, seed_astro=60,
        verbose=True,
    )

    # ------ Parameter search space (Report 2) -------------------------------
    Sigma_arr      = Real(2,    6,    name='Sigma')         # mV
    g_AHP_arr      = Real(1,    15,   name='g_AHP')         # nS
    Xi_ampa_arr    = Real(0.2,  1,    name='Xi_ampa')       # 1/mmole
    Xi_nmda_arr    = Real(0.2,  1,    name='Xi_nmda')       # 1/mmole
    Tau_Ca_arr     = Real(1,    11,   name='Tau_Ca')        # second
    U_0_ar_arr     = Real(0,    0.005, name='U_0_ar')
    U_max_arr      = Real(0.1,  1,    name='U_max')         # 1/ms
    U_0_sr_arr     = Real(0.1,  1,    name='U_0_sr')
    Omega_f_sr_arr = Real(0.1,  4.5,  name='Omega_f_sr')    # 1/second
    Omega_f_ar_arr = Real(0.1,  4.5,  name='Omega_f_ar')    # 1/second
    Omega_d_arr    = Real(0.1,  4.5,  name='Omega_d')       # 1/second
    alpha_syn_arr  = Real(0.1,  1,    name='alpha_syn')
    g_na_arr       = Real(0.5*50, 3*50, name='g_na')
    g_kd_arr       = Real(0.5*5,  3*5,  name='g_kd')

    space = [
        Sigma_arr, g_AHP_arr,
        Xi_ampa_arr, Xi_nmda_arr, Tau_Ca_arr,
        U_0_ar_arr, U_max_arr, U_0_sr_arr,
        Omega_f_sr_arr, Omega_f_ar_arr, Omega_d_arr,
        alpha_syn_arr,
        g_na_arr, g_kd_arr,
    ]

    N_SAMPLES = 1_000_000
    params = [
        list(i) for i in zip(*[d.rvs(n_samples=N_SAMPLES) for d in space])
    ]

    # ------ First sanity-check run with the nominal point -------------------
    initial_param_vector_numerical = np.array([
        4.0,        # Sigma (mV)
        5.0,        # g_AHP (nS)
        0.5,        # Xi_ampa (1/mmole)
        0.3,        # Xi_nmda (1/mmole)
        8.0,        # Tau_Ca (s)
        0.003,      # U_0_ar (unitless)
        0.5,        # U_max (1/ms)
        0.15,       # U_0_sr (unitless)
        2.0,        # Omega_f_sr (1/s)
        1.42857,    # Omega_f_ar (1/s, = 1/0.7)
        2.0,        # Omega_d (1/s)
        1.0,        # alpha_syn (unitless)
        80.0,       # g_na coefficient (1.6 * 50)
        6.5,        # g_kd coefficient (1.3 * 5)
    ])

    out_path = sim.do_run(
        initial_param_vector_numerical,
        iteration=0,
        seed_run=None,                                                    # paired-comparison mode
    )
    print(f'Sanity run -> {out_path}')

    # ------ Example: sweep loop ---------------------------------------------
    # for k, row in enumerate(params, start=1):
    #     sim.do_run(row, iteration=k, seed_run=None)
