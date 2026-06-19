from mpl_toolkits.mplot3d import Axes3D
from mpl_toolkits.mplot3d.art3d import Poly3DCollection, Line3DCollection
import matplotlib.colors as mcolors
from sklearn.metrics import mean_squared_error
import numpy as np
import scipy.io
import os
from scipy.io import loadmat
from scipy.signal import find_peaks
from scipy.ndimage import gaussian_filter1d
from scipy.integrate import odeint
from sklearn.decomposition import PCA

from brian2 import *
from brian2 import devices

from sklearn.neighbors import KDTree
from scipy.stats import skewnorm
from scipy.stats import chi2
import seaborn as sns
import math



'''
Version: cython/cpp_standalone friendly. Connections and positions are randomly
placed at network-construction time and then frozen.

The Binomial_fun custom Brian2 Function is defined in the calling script
(`Main_code_notebook.py` for the HPC, `Main_code_PRINCIPAL.py` locally) and
passed into `Neuronal_Network(..., Binomial_fun=...)`. The library only
*references* the name `Binomial_fun` in synapse equations; it does not declare
the function itself.
'''


# ------------------ UTILITY FUNCTIONS ------------------
def extract_synaptic_connections(input_folder):
    """
    Loads back the connectivity and namespace files written by
    `save_synaptic_connections`.

    Returns:
        dict with keys (when files are present):
            S_source, S_target, GJ_source, GJ_target, StoA_source, StoA_target
            S_x_syn, S_y_syn               (synapse bouton positions, in um)
            Neuron_state, Astrocyte_state  (NpzFile objects)

    AtoS is NOT loaded — it is reconstructed by swapping (i, j) of StoA at
    network-build time.

    Missing position files (S_x_syn.npy, S_y_syn.npy) raise a warning but do
    not abort: this preserves backward compatibility with connectivity folders
    generated before the position-save was added.
    """
    print(f"\n--- Loading data from: {input_folder} ---")
    loaded_data = {}

    if not os.path.isdir(input_folder):
        raise FileNotFoundError(f"Input folder not found at '{input_folder}'.")

    # 1. Connection (i, j) arrays — required.
    synapse_groups = ["S", "GJ", "StoA"]
    for name in synapse_groups:
        source_path = os.path.join(input_folder, f'{name}_source_units.npy')
        target_path = os.path.join(input_folder, f'{name}_target_units.npy')
        loaded_data[f'{name}_source'] = np.load(source_path)
        loaded_data[f'{name}_target'] = np.load(target_path)
        print(f"  Loaded {len(loaded_data[f'{name}_source'])} connections for group '{name}'.")

    # 2. Synapse positions — optional (backward compat).
    x_path = os.path.join(input_folder, 'S_x_syn.npy')
    y_path = os.path.join(input_folder, 'S_y_syn.npy')
    if os.path.isfile(x_path) and os.path.isfile(y_path):
        loaded_data['S_x_syn'] = np.load(x_path)
        loaded_data['S_y_syn'] = np.load(y_path)
        print(f"  Loaded synapse bouton positions ({len(loaded_data['S_x_syn'])} synapses).")
    else:
        print("  WARNING: S_x_syn.npy / S_y_syn.npy not found. "
              "S.x_syn and S.y_syn will be left at 0; "
              "any downstream code that needs them will be wrong.")

    # 3. State namespaces — optional but expected.
    for name in ["Neuron", "Astrocyte"]:
        state_path = os.path.join(input_folder, f'{name}_state_namespace.npz')
        if os.path.isfile(state_path):
            loaded_data[f'{name}_state'] = np.load(state_path)
            print(f"  Loaded state namespace for '{name}'.")
        else:
            print(f"  WARNING: state namespace not found for '{name}' ({state_path}).")

    print("--- Load complete ---")
    return loaded_data
def save_synaptic_connections(output_folder, S, GJ, StoA, neuron_group, astrocyte_group):
    """
    Saves the source (i) and target (j) unit indices for the relevant Brian2
    Synapses objects, the per-synapse position arrays (x_syn, y_syn), and the
    state-variable namespaces (`get_states()`) of the Neuron and Astrocyte
    groups.

    AtoS is intentionally NOT saved: its (i, j) pairs are exactly the swap of
    StoA's, and the HPC loader reconstructs them on the fly. Saving them would
    create two sources of truth and risk inconsistency.

    Files written:
        S_source_units.npy,    S_target_units.npy        (synapse i, j)
        S_x_syn.npy,           S_y_syn.npy               (synapse bouton positions in um)
        GJ_source_units.npy,   GJ_target_units.npy       (gap-junction i, j)
        StoA_source_units.npy, StoA_target_units.npy     (syn-to-astro i, j)
        Neuron_state_namespace.npz                       (NeuronGroup.get_states())
        Astrocyte_state_namespace.npz                    (Astrocyte.get_states())

    Args:
        output_folder (str): Destination directory.
        S, GJ, StoA: Brian2 Synapses objects.
        neuron_group, astrocyte_group: NeuronGroup objects.
    """
    print(f"--- Saving connections and namespaces to: {output_folder} ---")

    os.makedirs(output_folder, exist_ok=True)

    synapse_groups = {"S": S, "GJ": GJ, "StoA": StoA}

    for name, syn_group in synapse_groups.items():
        source_indices = np.array(syn_group.i)
        target_indices = np.array(syn_group.j)
        np.save(os.path.join(output_folder, f'{name}_source_units.npy'), source_indices)
        np.save(os.path.join(output_folder, f'{name}_target_units.npy'), target_indices)
        print(f"  Saved {len(source_indices)} connections for group '{name}'.")

    # Synapse bouton positions: required to reconstruct distance-based rules on the HPC.
    try:
        x_syn = np.array(S.x_syn / um)
        y_syn = np.array(S.y_syn / um)
        np.save(os.path.join(output_folder, 'S_x_syn.npy'), x_syn)
        np.save(os.path.join(output_folder, 'S_y_syn.npy'), y_syn)
        print(f"  Saved synapse bouton positions ({len(x_syn)} synapses).")
    except AttributeError:
        print("  WARNING: S has no x_syn/y_syn attributes — skipping position save.")

    # Per-group state-variable snapshots
    state_groups = {"Neuron": neuron_group, "Astrocyte": astrocyte_group}
    for name, group in state_groups.items():
        state_data = group.get_states()
        state_path = os.path.join(output_folder, f'{name}_state_namespace.npz')
        np.savez_compressed(state_path, **state_data)
        print(f"  Saved state namespace for '{name}'.")

    print("--- Save complete ---")

    
    

def plot_layered_connections_with_mea_planar(neurons, astrocytes, gj_synapses, neuron_synapses, Grid):
    """
    Plots the positions of neurons, astrocytes, and MEA electrodes (as planar discs)
    in separate Z-layers and visualizes connections in 3D.

    Args:
        neurons (NeuronGroup): The neuron population. Assumed to have .x, .y.
        astrocytes (NeuronGroup): The astrocyte population. Assumed to have .x_astro, .y_astro.
        gj_synapses (Synapses): The astrocyte-to-astrocyte Gap Junction connections.
        neuron_synapses (Synapses): The neuron-to-neuron connections.
        Grid (np.array): A (12, 2) array of (x, y) positions of the 12 MEA electrodes in um.
    """
    
    # --- Configuration and Unit Conversion ---
    
    Z_MEA_LAYER    = -1.0  # NEW: Bottom layer for electrodes
    Z_NEURON_LAYER = 0.0   # Middle layer
    Z_ASTRO_LAYER  = 1.5   # Top layer
    
    MEA_ELECTRODE_RADIUS = 40.0 # [um] The actual radius for drawing the circles
    NUM_CIRCLE_POINTS = 50     # Number of points to approximate a circle
    
    scale_factor = umeter

    # Astrocyte coordinates
    try:
        astro_x = astrocytes.x_astro / scale_factor
        astro_y = astrocytes.y_astro / scale_factor
    except AttributeError:
        astro_x = astrocytes.x_astro
        astro_y = astrocytes.y_astro
    astro_z = np.full_like(astro_x, Z_ASTRO_LAYER)

    # Neuron coordinates
    neuron_x = neurons.x / scale_factor
    neuron_y = neurons.y / scale_factor
    neuron_z = np.full_like(neuron_x, Z_NEURON_LAYER)
    
    # MEA Grid coordinates (already in um)
    mea_x = Grid[:, 0]
    mea_y = Grid[:, 1]
    
    # Get connection indices
    astro_pre_gj = gj_synapses.i
    astro_post_gj = gj_synapses.j
    neuron_pre_syn = neuron_synapses.i
    neuron_post_syn = neuron_synapses.j

    # --- Plotting Setup ---
    fig = plt.figure(figsize=(14, 12)) 
    ax = fig.add_subplot(111, projection='3d')
    ax.set_title('Neural, Glial, and MEA Networks in 3D', color='white', fontsize=16)
    
    # --- Background and Grid Aesthetics ---
    fig.patch.set_facecolor('#282c34') 
    ax.set_facecolor('#1e222a') 

    # Set pane colors for dark background
    pane_color = (0.1, 0.1, 0.1, 1.0)
    ax.xaxis.pane.set_color(pane_color)
    ax.yaxis.pane.set_color(pane_color)
    ax.zaxis.pane.set_color(pane_color)
    ax.xaxis.pane.set_edgecolor('w')
    ax.yaxis.pane.set_edgecolor('w')
    ax.zaxis.pane.set_edgecolor('w')

    ax.grid(True, linestyle=':', alpha=0.4, color='gray') 

    # ------------------------------------------------------------------
    # --- NEW LAYER: Plot MEA Electrodes as PLANAR DISCS (Layer Z=-0.5) ---
    # ------------------------------------------------------------------
    electrode_patches = []
    for i in range(len(mea_x)):
        # Generate points for a circle
        theta = np.linspace(0, 2*np.pi, NUM_CIRCLE_POINTS)
        x_circle = mea_x[i] + MEA_ELECTRODE_RADIUS * np.cos(theta)
        y_circle = mea_y[i] + MEA_ELECTRODE_RADIUS * np.sin(theta)
        z_circle = np.full_like(x_circle, Z_MEA_LAYER)
        
        # Create a polygon patch for the electrode (a filled circle)
        # Poly3DCollection expects a list of (N, 3) arrays, where N is the number of points for each polygon
        electrode_patches.append(list(zip(x_circle, y_circle, z_circle)))
    
    # Add all electrode patches to the plot
    collection = Poly3DCollection(electrode_patches, 
                                  facecolors='#808080',   # Gray fill
                                  edgecolors='white',     # White outline
                                  linewidths=1.0,
                                  alpha=0.6,
                                  zorder=1)
    ax.add_collection3d(collection)
    
    # Add a dummy point for the legend entry (Poly3DCollection doesn't automatically create one)
    ax.scatter([], [], [], # No data points
               s=100,      # Example size for legend marker
               c='#808080', 
               marker='o', 
               edgecolors='white', 
               linewidths=1.0,
               alpha=0.6,
               label=f'MEA Electrodes (Z={Z_MEA_LAYER} $\mu m$)', 
               zorder=1)


    # ------------------------------------------------------------------
    # --- 1. Plot Neurons (Layer Z=0.0) ---
    # ------------------------------------------------------------------
    ax.scatter(neuron_x, neuron_y, neuron_z, 
               s=50,      
               c='#00BFFF', 
               marker='D', 
               edgecolors='white', 
               linewidths=0.5,
               alpha=0.9,
               label=f'Neurons (Z={Z_NEURON_LAYER} $\mu m$)', 
               zorder=5) 

    # ------------------------------------------------------------------
    # --- 2. Plot Astrocytes (Layer Z=1.5) ---
    # ------------------------------------------------------------------
    ax.scatter(astro_x, astro_y, astro_z, 
               color='#FF4500', 
               marker='p',     
               s=90,          
               edgecolors='white', 
               linewidths=0.7,
               alpha=0.9,
               label=f'Astrocytes (Z={Z_ASTRO_LAYER} $\mu m$)', 
               zorder=6) 

    # --- 3. Plot Astrocyte-Astrocyte Gap Junctions (Within Astro Layer) ---
    for i in range(len(astro_pre_gj)):
        pre_idx = astro_pre_gj[i]
        post_idx = astro_post_gj[i]

        label = 'Astrocyte Gap Junction' if i == 0 else None
        
        ax.plot(
            [astro_x[pre_idx], astro_x[post_idx]],
            [astro_y[pre_idx], astro_y[post_idx]],
            [Z_ASTRO_LAYER, Z_ASTRO_LAYER], 
            color='#FFD700', 
            linestyle='-',
            alpha=0.8,       
            linewidth=3.0,   
            label=label
        )
    
    # --- 4. Plot Neuron-Neuron Synaptic Connections (Within Neuron Layer) ---
    for i in range(len(neuron_pre_syn)):
        pre_idx = neuron_pre_syn[i]
        post_idx = neuron_post_syn[i]

        label = 'Neuron-Neuron Synapse' if i == 0 else None
        
        ax.plot(
            [neuron_x[pre_idx], neuron_x[post_idx]],
            [neuron_y[pre_idx], neuron_y[post_idx]],
            [Z_NEURON_LAYER, Z_NEURON_LAYER], 
            color='#32CD32', 
            linestyle='--',
            alpha=0.2,       
            linewidth=1.0,   
            label=label
        )
        
    # --- Final Plot Aesthetics ---
    ax.set_xlabel('X position ($\mu m$)', color='white', fontsize=12)
    ax.set_ylabel('Y position ($\mu m$)', color='white', fontsize=12)
    ax.set_zlabel('Z position ($\mu m$)', color='white', fontsize=12)
    
    # Set axis tick colors to white
    ax.tick_params(axis='x', colors='white')
    ax.tick_params(axis='y', colors='white')
    ax.tick_params(axis='z', colors='white')
    
    # Set axis limits based on all coordinate data
    all_x = np.concatenate([neuron_x, astro_x, mea_x])
    all_y = np.concatenate([neuron_y, astro_y, mea_y])
    
    if all_x.size > 0:
        x_min, x_max = np.min(all_x), np.max(all_x)
        y_min, y_max = np.min(all_y), np.max(all_y)
        x_range = x_max - x_min
        y_range = y_max - y_min
        
        pad_x = x_range * 0.1
        pad_y = y_range * 0.1
        ax.set_xlim(x_min - pad_x, x_max + pad_x)
        ax.set_ylim(y_min - pad_y, y_max + pad_y)
    
    z_min = Z_MEA_LAYER - 0.5
    z_max = Z_ASTRO_LAYER + 0.5
    ax.set_zlim(z_min, z_max)
    
    # Legend with white text
    legend = ax.legend(loc='upper right', markerscale=1.5, fontsize=10, facecolor='#282c34', edgecolor='white')
    plt.setp(legend.get_texts(), color='white') 
    
    # Adjust view angle 
    ax.view_init(elev=30, azim=-70) 
    
    plt.tight_layout() 
    plt.show()


def Distance_based_connections(N, params_Syn, sed):
    """
    Build a distance-dependent adjacency matrix on the neuronal population.

    NOTE: this function is NOT currently wired into Neuronal_Network. The
    active synaptic-connection rule is the random one (`S.connect(p=conn_prob,
    condition='i!=j')`) followed by spatial repositioning of the bouton via
    `get_synapse_coordinates`. Kept here as a reference alternative that can
    be plugged back in by replacing the `connect(p=...)` call inside
    `Neuronal_Network`.

    The ADJ matrix convention is:
        rows: pre-synaptic neurons
        columns: post-synaptic neurons
    """
    # Use NumPy's random number generator for better performance
    rng = np.random.default_rng(sed)

    # Retrieve the number of neurons
    Nn = N.N

    # Extract x and y coordinates and scale them
    coords = np.squeeze(np.array([[n.x / um, n.y / um] for n in N]))

    # Calculate all pairwise distances at once using broadcasting
    # This creates a matrix where element (i, j) is the distance between neuron i and neuron j
    x_diff = coords[:, 0][:, np.newaxis] - coords[:, 0]
    y_diff = coords[:, 1][:, np.newaxis] - coords[:, 1]
    distances = np.sqrt(x_diff**2 + y_diff**2)

    # Calculate probabilities for all connections simultaneously
    probabilities = -distances * params_Syn['slope'] + params_Syn['intercept']

    # Generate a single matrix of random numbers
    random_matrix = rng.random((Nn, Nn))

    # Compare the random numbers to the probabilities to determine connections
    # This creates a boolean array, which is then converted to integers (0s and 1s)
    ADJ = (random_matrix < probabilities).astype(int)

    # Remove self-connections by setting the diagonal to zero
    np.fill_diagonal(ADJ, 0)

    return ADJ


def weibull_connections(N_pos, p0, d0, beta, c_max, seed,
                        periodic=False, chunk=2048):
    """
    Directed neuron->neuron edge list under the Weibull / stretched-exponential
    distance kernel

        p(d) = p0 * exp( -(d / d0)**beta ),     d in micrometres.

    This is the canonical implementation of the distance-dependent wiring rule;
    both the single-run driver and the sweep driver call it, so the rule lives
    in exactly one place. It is the distance-dependent counterpart of the flat
    Bernoulli rule used in build_topology, and returns the SAME object an edge
    list, with the SAME index convention (i = pre = row, j = post = column),
    so the bouton-placement and S.connect(i=Source, j=Target) code downstream is
    agnostic to how the edges were produced.

    Parameters
    ----------
    N_pos : (Nn, 2) float ndarray
        Neuron (x, y) positions in micrometres.
    p0 : float in (0, 1]
        Connection probability at zero distance, p(0).
    d0 : float [um], > 0
        Characteristic length scale of the kernel.
    beta : float, > 0
        Stretch exponent. beta == 1 is a plain exponential; beta < 1 gives a
        heavier-than-exponential tail.
    c_max : float [um]
        Arena side length. Used ONLY when periodic=True (sets the wrap period).
    seed : int
        RNG seed for the per-pair Bernoulli draws. By convention the caller
        passes seed_synapse, keeping synapse-wiring randomness on one stream.
    periodic : bool, optional
        If True, use the minimum-image (flat-torus) distance
            dx <- dx - c_max * round(dx / c_max)     (and likewise dy)
        so every neuron sees an identical, edge-free neighbourhood. If False
        (default) use ordinary bounded Euclidean distance on [0, c_max]^2.
    chunk : int, optional
        Row block size; bounds peak memory to O(chunk * Nn) rather than
        O(Nn**2) for the pairwise-distance matrix at large Nn.

    Returns
    -------
    S_i, S_j : (n_syn,) int32 ndarrays
        Presynaptic (row) / postsynaptic (column) indices. Autapses (i == j)
        are excluded by construction.
    """
    rng = np.random.default_rng(seed)
    Nn = N_pos.shape[0]
    px = N_pos[:, 0]
    py = N_pos[:, 1]

    ii_list, jj_list = [], []
    for r0 in range(0, Nn, chunk):
        r1 = min(r0 + chunk, Nn)
        dx = px[r0:r1][:, None] - px[None, :]          # (B, Nn) row-pre/col-post
        dy = py[r0:r1][:, None] - py[None, :]
        if periodic:                                   # minimum-image wrap
            dx -= c_max * np.round(dx / c_max)
            dy -= c_max * np.round(dy / c_max)
        d = np.sqrt(dx * dx + dy * dy)
        P = p0 * np.exp(-((d / d0) ** beta))           # <= 1 since p0 <= 1
        # Forbid autapses: zero the (global) diagonal entries in this row block.
        local_rows = np.arange(r1 - r0)
        P[local_rows, np.arange(r0, r1)] = 0.0
        hit = rng.random(P.shape) < P
        bi, bj = np.where(hit)
        ii_list.append(bi + r0)                        # map block-local row -> global
        jj_list.append(bj)

    S_i = np.concatenate(ii_list).astype(np.int32) if ii_list else np.empty(0, np.int32)
    S_j = np.concatenate(jj_list).astype(np.int32) if jj_list else np.empty(0, np.int32)
    return S_i, S_j


def normalize_to_range(data, min_old, max_old, min_new, max_new):
    """
    Normalizes data from an old range [min_old, max_old]
    to a new range [min_new, max_new].

    Args:
        data (numpy.ndarray or list): The input data to normalize.
        min_old (float): The minimum value of the original data range.
        max_old (float): The maximum value of the original data range.
        min_new (float): The desired minimum value of the normalized data.
        max_new (float): The desired maximum value of the normalized data.

    Returns:
        numpy.ndarray: The normalized data.
    """
    # Handle the case where min_old == max_old to avoid division by zero
    if max_old == min_old:
        # If all data points are the same, they should all map to the midpoint of the new range
        # Or, if you want them all to be min_new, use min_new directly.
        # Here, we map to the midpoint.
        return np.full_like(data, (min_new + max_new) / 2.0, dtype=float)

    # Convert data to numpy array for consistent operations
    data = np.asarray(data, dtype=float)

    # Apply the normalization formula
    normalized_data = (data - min_old) * ((max_new - min_new) / (max_old - min_old)) + min_new
    return normalized_data


def unfold_ADJ(ADJ):
    
    '''
    This function is meant to take in input an adejency matrix (not necessarily squared)
    and return the source and target units' index.
    
    The ADJ matrix must comply to the following concention:
        rows    : pre
        columns : post
    
    '''
    # Extract dimensions
    rows, cols = ADJ.shape
    
    
    # Initialize variables
    Source = []
    Target = []
    
    
    
    # Run across rows (pre units)
    for pre in range(rows):
        
        # Extract indicies
        Post_idx = np.where( ADJ[pre,:] != 0 )[0]
        
        if Post_idx.size == 0: # Empty array....no post units
            continue
        
        # Construct pre vector
        source_num = np.full(len(Post_idx), pre)
        
        # Update variables 
        Source.append(source_num)
        Target.append(Post_idx)
        
        
        
        
        
    return np.concatenate(Source), np.concatenate(Target)
        









def get_Astroparam(oscillations = 'AM',**kwargs):
    
    
    params = {
        # ----Input
        'f_in': 1.*Hz,              # Input frequency (synapse)
        'f_c' : 1.*Hz,              # Input frequency (gliotransmission)
        # 't_on' : 0*second,         # Start of synaptic stimulation (used in STDP)
        # 't_off' : Inf*second,      # End of astrocyte stimulation (used in standalone gliotransmission)
        # --- IP_3R kinectics
        'd_1': 0.13*umole,         # IP_3 binding affinity
        'O_2': 0.2/umole/second,   # Inactivating Ca^2+ binding rate
        'd_2': 1.05*umole,         # Inactivating Ca^2+ binding affinity
        'd_3': 0.9434*umole,       # IP_3 binding affinity (with Ca^2+ inactivation)
        'd_5': 0.08*umole,         # Activating Ca^2+ binding affinity
        # ---  Calcium fluxes
        'C_osc': 0.2*umole,        # Estimated Threshold for Ca^2+ oscillations
        'C_T': 2*umole,            # Total ER Ca^2+ content
        'rho_A': 0.18,             # ER-to-cytoplasm volume ratio
        'Omega_C': 6/second,       # Maximal Ca^2+ release rate by IP_3Rs
        'Omega_L': 0.1/second,     # Maximal Ca^2+ leak rate,
        'O_P': 0.9*umole/second,   # Maximal Ca^2+ uptake rate
        # K_P (see below)          # Ca^2+ affinity of SERCA pumps
        # --- IP_3 production
        # Omega_delta (see below)  # Maximal rate of IP_3 production by PLCdelta
        'K_delta': 0.5*umole,      # Ca^2+ affinity of PLCdelta
        'kappa_delta': 1.*umole,    # Inhibiting IP_3 affinity of PLCdelta
        # --- IP_3 degradation
        # Omega_5P (see below)     # Maximal rate of IP_3 degradation by IP-5P
        'O_3K': 4.5*umole/second,  # Maximal rate of IP_3 degradation by IP_3-3K
        'K_D': 0.5*umole,          # Ca^2+ affinity of IP3-3K
        'K_3K': 1.*umole,           # IP_3 affinity of IP_3-3K
        # --- IP_3 diffusion
        'F': 2.*umole/second,       # GJC IP_3 permeability (nonlinear)
        'I_Theta': 0.3*umole,      # Threshold IP_3 gradient for diffusion
        'omega_I': 0.05*umole,     # Scaling factor of diffusion
        # Base fallbacks so per-astrocyte IC writes (Astro.Omega_5P / Astro.I_bias)
        # work in ALL modes, not only AM/FM. The AM/FM branches below still override
        # these with mode-specific values. Defaults here are the AM nominals.
        'Omega_5P': 0.1/second,    # IP_3-5P degradation rate (base fallback)
        'I_bias': 0.8*umole,       # IP_3 exogenous set-point (base fallback)
        # --- Agonist-dependent IP_3 production
        'O_beta': 1.*umole/second,  # Maximal rate of IP_3 production by PLCbeta
        'O_N': 0.3/umole/second,   # Agonist binding rate
        'Omega_N': 1.8/second,     # Inactivation rate of GPCR signalling
        'K_KC': 0.5*umole,         # Ca^2+ affinity of PKC
        'zeta': 2.,                # Maximal reduction of receptor affinity by PKC
        'n': 1.,                   # Cooperativity of agonist binding reaction
        # --- Gliotransmitter release and time course        
        'C_Theta': 0.5*umole,      # Ca^2+ threshold for exocytosis
        'Omega_A': 0.6/second,     # Gliotransmitter recycling rate
        'U_A': 0.6,                # Gliotransmitter release probability
        'G_T': 200.*mmole,         # Total vesicular gliotransmitter
        'rho_e': 6.5e-4,           # Ratio of astrocytic vesicle volume/ESS volume
        'Omega_e': 5./second,      # Gliotransmitter clearance rate (think about distributed release)
        'spill_over': 0.75,         # Spill over parameter
        
        # Connection probability
        'conn_dist' : 200, # [um] — legacy 'distance' rule radius
        'gj_max_dist': 150, # [um] — soft cap for the 'wallach' Voronoi rule
        'c_min' : 0, #[um]
        'c_max' : 1100, # [um]
        
        # Stimulation
        
      
    }

    if oscillations == 'AM':
        params.update({
            'K_P': 0.1*umole,
            'O_delta': 0.01*umole/second,
            'Omega_5P': 0.1/second,
            'I_bias': 0.8*umole
        })
    elif oscillations == 'FM':
        params.update({
            'K_P': 0.05*umole,
            'O_delta': 0.05*umole/second,
            'Omega_5P': 0.1/second,
            'I_bias': 1.*umole
        })
    
    
    return params
    
    

def get_Neuronparam(**kwargs):
    
    
    
    Neuron_area = 300*umetre**2

    # ── CAdEx (Górski et al. 2021) — phenomenological replacement of the HH model ──
    # Paper convention: absolute units (pF / nS / mV). The 10 swept intrinsic axes
    # below are STANDALONE-run fallbacks; at sweep runtime each is shadowed per
    # neuron by run_args (HPC_main_sweep / HPC_single_run).
    params = {
    'area': Neuron_area,            # kept as metadata; NOT used by the CAdEx eqs

    # --- Fixed CAdEx constants (namespace) ---
    'Cm':     200*pF,               # membrane capacitance C
    'El':    -55*mV,                # leak reversal E_L (depolarized → HH-like 5 mV rest-to-threshold gap)
    'EA':    -70*mV,                # adaptation reversal E_A
    'VD':    -40*mV,                # spike detection / reset trigger (paper cutoff)
    't_ref':   3*ms,                # refractory period
    'gl_ref':  10*nS,               # FIXED leak ref for the diffusion noise
                                    #   (decoupled from the swept per-neuron gl)

    # --- Swept CAdEx per-neuron axes (defaults; shadowed by run_args) ---
    'sigma':    4*mV,               # idx 0   noise amplitude
    'gbarA':   10*nS,               # idx 1   max subthreshold adaptation conductance ḡ_A
    'tauA':   200*ms,               # idx 4   adaptation time constant τ_A
    'DeltaT':   2*mV,               # idx 12  spike-initiation slope Δ_T
    'VT':     -48*mV,               # idx 13  spike threshold V_T (frozen)
    'delta_gA': 1*nS,               # idx 16  post-spike adaptation increment δg_A
    'gl':      10*nS,               # idx 30  leak conductance g_L
    'VA':     -45*mV,               # idx 31  subthreshold adaptation activation V_A
    'DeltaA':   5*mV,               # idx 32  subthreshold adaptation slope Δ_A (> 0)
    'VR':     -55*mV,               # idx 33  reset potential V_R

    # --- Drive & synaptic (unchanged) ---
    'I_inj':  15*pA,                # per-neuron bias scale (N.I = (rand-0.5)*I_inj)
    'E_ampa':  0*mV,
    'E_nmda':  0*mV,
    'g_ampa': 1.6*nS,               # idx 14  (synaptic conductance, set on the neuron)
    'g_nmda': 0.4*nS,               # idx 15

    # --- Position ---
    'c_min' : 0,                    # [µm]
    'c_max' : 1100,                 # [µm]
     }
    
    
    params.update(kwargs)

    return params






def get_Synparam(synapse_type='depressing',**kwargs):
    
    params = {
        
        'area' : 300*umetre**2,  
        # --- Synaptic dynamics
        'E_Iper': 80/100,          # Persentage of excitatory connections
        # Omega_d (see below)      # Depression rate
        # Omega_f (see below)      # Facilitation rate,
        # U_0_sr (see below)    # Basal synaptic release probability
        'Omega_c': 500./second,    # Fast intra-cleft clearance (tau_clear = 2 ms) -> drives receptors
        'Omega_Aclear': 33.3/second, # Slow spillover clearance (tau_A = 30 ms) -> drives astrocyte
                                     # NOTE: renamed from spec's 'Omega_A' to avoid collision with the
                                     # gliotransmitter recycling rate 'Omega_A' (0.6/s) in get_Astroparam().
        'rho': 0.005,            # synaptic vesicle-to-extracellular space volume ratio
        'Y_T': 6000.*mmole,        # Total resource = 12 docked vesicles x 500 mM (Kusick 2020)
        # --- Presynaptic receptors
        'O_G': 1.5/umole/second,   # Agonist binding rate (activating)
        'Omega_G': 0.5/(60*second),# Agonist release rate (inactivating)
        # alpha_syn (see below)        # Gliotransmitter effect on synaptic release
        # --- SIC/SOC
        'G_sic'     : 4.5*mV,      # Max SIC/SOC depolarization
        'tau_sic_r' : 30.*ms,      # SIC/SOC rise time constant
        'tau_sic' : 600.*ms,       # SIC/SOC decay time constant
        
       # (Removed: tau_rise_NT / tau_decay_NT — these were the Y_S double-exponential
       #  time constants used only by the former Decay_type=='Double_exp' branch, which
       #  the fast/slow cleft split replaced. No longer referenced anywhere.)
        
       
    
       # Synapse parameters (uncommented and added to dictionary)
       # If these are meant to be included, they should also be added as key-value pairs

        'tau_ampa': 2 * ms,
        'taus_nmda': 100 * ms, # Decay
        'taux_nmda': 2 * ms, # Rise
        'tau_ampa_std': 0.02 * ms,
        'taus_nmda_std': 10 * ms,
        'taux_nmda_std': 0.02 * ms,
        'alpha_nmda': 0.5 * kHz,
        'tau_d': 200 * ms,
        'U': 0.2,
        'STF': False,
        'tau_f': 1000 * ms,
        
        'w':1,

        # Params of the kinetic model post-syn
        'tau_rise_ampa': 1*ms,
        'tau_decay_ampa': 2*ms,
        'tau_rise_nmda': 2*ms,
        'tau_decay_nmda': 100*ms,
        
        # Postsynaptic Hill dose-response (Pankratov & Krishtal 2003), kappa/mu-reconciled
        # to the model's rho*C_ves = 2.5 mM/vesicle convention (raw P&K /3.17).
        'EC50_ampa': 8.6*mmole,    # AMPA half-activation  (raw P&K 27.2 mM at B_tot=0.5)
        'EC50_nmda': 3.0*mmole,    # NMDA half-activation  (raw P&K  9.5 mM at B_tot=0.5)
        'n_ampa':    1.4,          # AMPA Hill exponent (P&K range 1.3-1.6)
        'n_nmda':    1.9,          # NMDA Hill exponent (P&K range 1.7-2.1)
        
        # Connection probability (flat / distance-independent rule)
        'conn_prob' : 0.107, # Random 
        
        # Distance dependent (legacy linear rule -- kept for reference)
        'slope': 1/500, # in [um] sHOULD BE 500
        'intercept': 0.8, # 1 usually

        # Distance dependent (Weibull / stretched-exponential kernel)
        #   p(d) = p0_conn * exp( -(d / d0_conn)**beta_conn ),  d in micrometres.
        # Swept (in the OUTER topology loop, NOT run_args) when --conn_rule weibull.
        # NOTE the '_conn' suffix is mandatory: the bare keys 'beta' (Graupner-
        # Brunel STDP coefficient, see below in this dict) and 'rho' (vesicle
        # volume ratio) are ALREADY taken; reusing them would silently overwrite
        # an unrelated synaptic coefficient.
        'p0_conn'  : 0.6,   # [dimensionless]  p(0); connection prob at zero distance
        'd0_conn'  : 100.0, # [um]             characteristic length scale
        'beta_conn': 1.0,   # [dimensionless]  stretch exponent (1 == plain exponential)
        
        # Connection to astrocyte rules
        'Conn_syn_astro_cutoff' :70 *um,
        'sigma_A': 200*um,   #150*um,

       
    
       # Asynchronous Release parameters (uncommented and added to dictionary)
 
       # 'Omega_f_ar': 1/ (0.7 * second),
       # 'U_0_ar': 0.003, #0.003
       # 'Umax': 0.5/ms,
       'x0': 0.2, # x0 seems to be unitless here
    
    }
    


    # ------------------ SYNAPSES ------------------
    if synapse_type == 'depressing':
        params.update({
            # 'Omega_d': 2./second,
            # 'Omega_f_sr': 3.33/second,
            # 'U_0_sr': 0.6,
            # 'alpha_syn': 0.,
        })
    elif synapse_type == 'facilitating':
        params.update({
            # 'Omega_d': 2./second, #2
            # 'Omega_f_sr': 2./second,
            # 'U_0_sr': 0.15,
            # 'alpha_syn': 1.       #1.,  
        })
    elif synapse_type == 'neutral':
        params.update({
            # 'Omega_d': 3./second,
            # 'Omega_f_sr': 3./second,
            # 'U_0_sr': 0.5,
            # 'alpha_syn': 1.,
        })
    else:
        raise ValueError('synapse argument has to be "depressing", "facilitating" or "neutral"')
    
   
    # parameters.update({
    # 'G_norm'     : normalize(1.0,parameters['tau_e_r'],parameters['tau_e']),
    # 'G_sic_norm' : normalize(1.0,parameters['tau_sic_r'],parameters['tau_sic_d'])
    # })
    # STDP parameters
    # Graupner and Brunel (PNAS 2012) / DP curve
    params.update({
        # 'tau_ca': 20.0*ms, # Intrasynaptic Ca2+ decay constant
        'Cpre'  : 1.0,     # Presynaptic Ca2+ increase per spk
        'Cpost' : 2.0,     # Postynaptic Ca2+ increase per spk
        'Theta_d' : 1.0,   # LTD threshold
        'Theta_p': 1.3,    # LTP threshold
        'gamma_d': 200.0,  # LTD learning rate
        'gamma_p': 321.808,# LTP learning rate
        'W_0'    : 0.5,    # LTP/LTD boundary
        'tau_w'  : 346.3615*second, # Time decay of synaptic weights
        'D'      : 13.7*ms,# Synaptic delay
        # 'sigma_'  : 2.8284, # variance in the diffusion approx,
        'beta'   : 0.5,
        'b'      : 5.
    })
    

    
    params.update(kwargs)

    return params





def Neuronal_Network(Nn, Syn_pdist=None, ics=False, Simulated_network='Neuronal',
                     Decay_type='Double_exp', synapse_type='neutral', conn_prob_=None,
                     seed_neu=None, seed_syn=None, connections=None,
                     Binomial_fun=None, syn_positions=None):
    """
    Build the neuronal `NeuronGroup` ('Neuron') and synaptic `Synapses` group
    ('Synapse').

    Parameters
    ----------
    Nn : int
        Number of neurons.
    Syn_pdist : pandas.DataFrame or None
        Two-column table ('Syn_prob', 'Radius_val') giving the dendritic-arbour
        PDF used to place bouton positions when `connections=True`. Required
        only in that branch.
    ics : {False, 'rand'}
        If 'rand', overrides default initial conditions with random values
        (currently dormant — kept for future use, see Report 3 §B.1).
    Simulated_network : {'Neuronal', 'Full', 'Astrocytic'}
        Selects whether the synapse model includes the gliotransmission branch.
    Decay_type : {'Single_exp', 'Double_exp'}
        Extracellular glutamate decay kinetics.
    synapse_type : {'depressing', 'facilitating', 'neutral'}
        Tag forwarded to `get_Synparam`.
    conn_prob_ : float or None
        Overrides `conn_prob` in the synapse namespace.
    seed_neu, seed_syn : int or None
        Seeds for neuron-placement RNG and synapse-bouton-placement RNG.
        `seed_syn` is critical for reproducibility of synapse positions when
        `connections=True`.
    connections : {None, True, list}
        - None: synapses are created but not connected.
        - True: random rule with distance-based bouton placement.
        - list [Source, Target]: explicit (i, j) arrays — the only form that
          survives cpp_standalone. Used by the HPC loader.
    Binomial_fun : brian2.Function
        Custom Function implementing a binomial draw. Required: the synapse
        equations reference `Binomial_fun(...)` by name.
    syn_positions : (N, 2) ndarray or None
        Pre-computed bouton positions in um, paired with the (i, j) of
        `connections`. Used by the HPC loader to restore S.x_syn, S.y_syn.

    Returns
    -------
    N : NeuronGroup
    S : Synapses
    """

    if Binomial_fun is None:
        raise ValueError(
            "Neuronal_Network requires a Brian2 Function 'Binomial_fun' "
            "passed via the Binomial_fun= argument."
        )

    # ---------------------- NEURONAL GROUP ----------------------
     # neuron model
    eqs_NN = Equations('''

    # ── CAdEx membrane dynamics (Górski et al. 2021, eqs 2.1-2.2) ──
    # leak + exponential spike-initiation + conductance-based adaptation g_A.
    # `noise`, the per-neuron quenched bias `I_inj*bias_unit` (run-time-swept
    # SCALE I_inj × build-frozen unit draw bias_unit ∈ [-0.5, 0.5]), and the
    # synaptic current `I_syn` enter the membrane balance. E[bias_unit]=0 ⇒ the
    # bias adds across-cell DISPERSION only, never net drive (anchored by E_L).
    dV/dt  = noise + (gl*(El-V) + gl*DeltaT*exp((V-VT)/DeltaT) + gA*(EA-V) + I_inj*bias_unit - I_syn)/Cm : volt
    dgA/dt = (gbarA/(1 + exp((VA-V)/DeltaA)) - gA)/tauA                                     : siemens

    # Diffusion (current-noise) term — amplitude pinned to a FIXED leak ref
    # gl_ref so it does NOT co-vary with the swept per-neuron gl.
    noise = sigma*(2*gl_ref/Cm)**.5*randn()/sqrt(dt) : volt/second (constant over dt)

    # Intrinsic transmembrane ionic current for the LFP/electrode point-source
    # path (Electrode_trace); excludes capacitive, I and I_syn — the CAdEx
    # analogue of the former HH I_cell.
    I_cell = gl*(El-V) + gl*DeltaT*exp((V-VT)/DeltaT) + gA*(EA-V) : amp

    x : meter
    y : meter

    # State Variables (Brian2 per-neuron parameters; swept via run_args)
    sigma    : volt
    gl       : siemens
    gbarA    : siemens
    delta_gA : siemens
    tauA     : second
    DeltaT   : volt
    VT       : volt
    VA       : volt
    DeltaA   : volt
    VR       : volt
    g_ampa   : siemens
    g_nmda   : siemens
    I_inj    : amp                # swept per-neuron bias SCALE (homogeneous value via run_args)
    bias_unit: 1                  # FROZEN per-neuron unit draw in [-0.5, 0.5], set at build

    ''')
    
    
    #%
    # ---------------------- SYNAPSES ----------------------
    '''
    
    # If a variable should be taken as a parameter of the neurons, 
    # i.e. if it should be possible to vary its value across neurons, 
    # it has to be declared as part of the model description:
    
        
    
    '''
    
    # -------------- Equations --------------
    
    # Synapses modelled as in Tsodyks (2005) with basal release probability
    # modulated by presynaptic receptors as in De Pitta' et al., PLoS Comput. Biol. (2011)
    #
    # IMPORTANT: 'postc' argument stands for 'post' in other methods, but because 'post' is a protected keyword in Synapse
    # it cannot be used in this module and 'postc' is used instead.
    
  
   
    
    # ----------------- SYNAPTIC EQUATIONS ----------------------
    
    # Basic synaptic equations are the same. what changes is how the post synaptic current behaves.
    
    
    
    
    if Simulated_network == 'Neuronal':

        # r_Ar (synonyms r_Sr) is the asynchronous (synchronous) release rate.
        # Units idiom: Binomial_fun(N, p) returns the integer number of vesicles
        # released during this time step. Dividing by dt makes it a rate (Hz).
        # When multiplied by dt in the integrator, it gives back the integer
        # per-step depletion of x_S — so r_Ar carries Hz but the discrete update
        # x_S(t+dt) -= x0 * Binomial_fun(...) is exact.

        eqs_Syn = Equations('''


            # Available neurotransmitter
            dx_S/dt = Omega_d * (1 - x_S) -  r_Ar: 1 (clock-driven)

            # Usage of releasable neurotransmitter per single action potential (synchronous):
            dusr/dt = -Omega_f_sr * usr : 1 (clock-driven)


            # Asynchronous release (vesicle counts per dt converted to a rate, see comment above)
            r_Ar = x0*nar : Hz
            nar = Binomial_fun(int(floor(x_S/x0)),uar*dt)/dt :Hz (constant over dt)
            duar/dt = -uar*Omega_f_ar :Hz (clock-driven)

            r_Sr : 1

            # Astrocyte ID for connection
            astro_index : integer

            # Positions
            x_syn : metre
            y_syn : metre


            # State Variables (Brian2 per-synapse parameters)
            Omega_d : 1/second
            Omega_f_sr : 1/second
            Omega_f_ar : 1/second
            U_0_sr : 1
            U_0_ar : 1
            Umax : 1/second
            alpha_syn : 1
            x0 : 1
            Y_T : mole
            O_G : 1/mole/second
            Omega_G : 1/second

            ''')
        
        # -------------- Event based update --------------
        
        
        pre = '''
        
            U_0 =  U_0_sr
            usr += U_0 * (1 - usr)
            r_Sr = usr * x_S # synchronously released synaptic neurotransmitter resources
            x_S -= r_Sr     
            uar += U_0_ar*(Umax-uar)
            '''
        post = None
        
        
       
           
        
        
    else:

        # Same units idiom as in the 'Neuronal' branch: r_Ar carries Hz but the
        # per-step decrement of x_S is exactly x0 * Binomial_fun(...).

        eqs_Syn = Equations('''
            # Fraction of activated presynaptic receptors
            dGamma_S/dt = O_G * G_A_syn * (1 - Gamma_S) - Omega_G * Gamma_S : 1 (clock-driven)


            # Available neurotransmitter
            dx_S/dt = Omega_d *(1 - x_S) -  r_Ar: 1 (clock-driven)

            # Usage of releasable neurotransmitter per single action potential (synchronous):
            dusr/dt = -Omega_f_sr * usr : 1 (clock-driven)


            # Asynchronous release
            r_Ar = x0*nar : Hz
            nar = Binomial_fun(int(floor(x_S/x0)),uar*dt)/dt :Hz (constant over dt)
            duar/dt = -uar*Omega_f_ar :Hz (clock-driven)


            # Define the variables of the model
            G_A_syn : mole  # gliotransmitter concentration in the extracellular space
            r_Sr : 1


            # Astrocyte ID for connection
            astro_index : integer

            # Positions
            x_syn : metre
            y_syn : metre

            # State Variables (Brian2 per-synapse parameters)
            Omega_d : 1/second
            Omega_f_sr : 1/second
            Omega_f_ar : 1/second
            U_0_sr : 1
            U_0_ar : 1
            Umax : 1/second
            alpha_syn : 1
            x0 : 1
            Y_T : mole
            O_G : 1/mole/second
            Omega_G : 1/second


            ''')
        
        # -------------- Event based update --------------
    
        pre = '''
        
            U_0 =  (1 - Gamma_S) * U_0_sr + alpha_syn * Gamma_S
            
            usr += U_0 * (1 - usr)
            r_Sr = usr * x_S # synchronously released synaptic neurotransmitter resources
            x_S -= r_Sr     
            uar += U_0_ar*(Umax-uar)
            
        '''
        post = None
        
        
      
         
        
        
    # ---------- Extrasyn glutamate model ----------
    
    # ---- Cleft glutamate: fast Y_S (receptors) + slow Y_A (astrocyte spillover) ----
    # Decay_type is retained as an argument for API compatibility but no longer
    # selects a kinetic scheme; both variables are single-exponential with
    # different time constants (Omega_c fast, Omega_Aclear slow).
    eqs_Syn += Equations('''
                         dY_S/dt = -Omega_c * Y_S + rho * Y_T * r_Ar       : mole (clock-driven)
                         dY_A/dt = -Omega_Aclear * Y_A + rho * Y_T * r_Ar  : mole (clock-driven)
                         ''')
    pre += '''
            Y_S += rho * Y_T * r_Sr
            Y_A += rho * Y_T * r_Sr
            '''


    
    # ----------- SYNAPTIC CURRENTS MODEL -------------
    
    
    eqs_Syn += Equations('''
                            # Hill activation targets (dimensionless, in [0,1]) of cleft glutamate Y_S
                            H_ampa = (Y_S/EC50_ampa)**n_ampa / (1 + (Y_S/EC50_ampa)**n_ampa) : 1
                            H_nmda = (Y_S/EC50_nmda)**n_nmda / (1 + (Y_S/EC50_nmda)**n_nmda) : 1

                            # AMPA: single-exponential, bounded saturating kick applied in pre
                            dr_ampa/dt = -r_ampa/tau_decay_ampa : 1 (clock-driven)

                            # NMDA: two-state cascade. x_r_nmda = slow reservoir (decay),
                            # r_nmda = fast follower (rise). Both bounded in [0,1].
                            dx_r_nmda/dt = -x_r_nmda/tau_decay_nmda                : 1 (clock-driven)
                            dr_nmda/dt   = (x_r_nmda - r_nmda)/tau_rise_nmda       : 1 (clock-driven)

                            r_ampa_tot_post = r_ampa : 1 (summed)
                            r_nmda_tot_post = r_nmda : 1 (summed)

                            # Per-synapse Hill parameters
                            EC50_ampa : mole
                            EC50_nmda : mole
                            n_ampa    : 1
                            n_nmda    : 1
                        ''')
                        
                        
    # NOTE: the H_ampa/H_nmda subexpressions (defined in eqs_Syn) evaluate to 0
    # when referenced by name inside on_pre in Brian2 2.10.1 (lazy/clock-phase
    # materialisation, confirmed by the acceptance test). Per spec section 4 fallback,
    # the Hill kick is inlined explicitly here so it reads the post-release Y_S.
    # H_ampa/H_nmda remain in eqs_Syn for continuous monitoring/diagnostics only.
    pre += '''
           r_ampa   += ((Y_S/EC50_ampa)**n_ampa/(1+(Y_S/EC50_ampa)**n_ampa)) * (1 - r_ampa)
           x_r_nmda += ((Y_S/EC50_nmda)**n_nmda/(1+(Y_S/EC50_nmda)**n_nmda)) * (1 - x_r_nmda)
           '''

    
    
    eqs_NN += Equations(''' 
                        I_syn =  I_ampa + I_nmda: amp
                        I_ampa = g_ampa*(V-E_ampa)*(r_ampa_tot) : amp
                        I_nmda = g_nmda*(V-E_nmda)*(r_nmda_tot)/(1+exp(-0.062*V/mV)/3.57) : amp
                        r_nmda_tot :1
                        r_ampa_tot :1
                        
                    
                        ''')

    
    
    
    # ----------- SYNAPTIC PARAMETERS ------------
    params_Syn = get_Synparam(synapse_type=synapse_type,conn_prob=conn_prob_)
    
    
    
    # -------------- Currents --------------
    
    
    # -------------------------- INITIALIZE THE NETWORKS ---------------------------
    
    # ---- Get parameters ----
    params_NN = get_Neuronparam()

    # Subthreshold span for the randomised V(0), injected as a namespace CONSTANT so
    # the V-init string 'El + rand() * V0_span' resolves El and V0_span from the
    # namespace and is NOT shadowed by the per-neuron VT state variable.
    # CRITICAL: at the point the IC 'N.V = ...' is evaluated, VT (state var) is still
    # 0 (it is set below). Using the namespace constant V0_span = VT_nominal - El
    # = 7.2 mV is therefore the only safe approach; any expression referencing
    # the state 'VT' directly would depolarise ~half the population to near 0 mV.
    params_NN['V0_span'] = params_NN['VT'] - params_NN['El']   # 7.2 mV at nominal (GROUNDED) op-point

    
    # Suppress Brian2's resolution_conflict warnings: every per-neuron state variable
    # declared in eqs_NN (sigma, gl, VT, I_inj, ...) also appears as a key in the
    # params_NN namespace (fallback for standalone runs). Brian2 correctly chooses
    # the internal state variable, so the warnings are expected and harmless -- but
    # they flood the log. This one call silences them globally for the process.
    BrianLogger.suppress_name('resolution_conflict')

    # CAdEx: V is detected/reset at VD (paper's -40 mV cutoff), reset to VR, and
    # the adaptation conductance is incremented by delta_gA. dtype=float64 (not
    # float32): the exp((V-VT)/DeltaT) term overflows float32 for small DeltaT /
    # low VT *below* the VD cutoff, whereas float64 has headroom to V≈0 mV. The
    # heavy Synapses group stays float32; the ~115-neuron group at float64 is free.
    N = NeuronGroup(Nn, model=eqs_NN, name='Neuron', namespace=params_NN,
                    threshold='V > VD', reset='V = VR; gA += delta_gA',
                    refractory=params_NN['t_ref'],
                    method='exponential_euler', dtype=float64)

    # Initial conditions
    # Randomise V(0) uniformly across [El, VT) to eliminate the synchronous startup volley.
    # With every cell previously pinned at El = -58.2 mV (7.2 mV below VT = -51.0 mV), the
    # entire population crossed threshold together at t ≈ 0 → one synchronous spike per neuron,
    # then silence on the fast (τ_m = 3.3 ms) membrane.  This was the source of the 19 %
    # one-spike-artifact fraction observed in campaign rv3.
    # Spreading V(0) uniformly across the subthreshold band desynchronises the first wave;
    # the onset transient shrinks to a short (<1 s) settling period that is easily discarded
    # in the feature extractor (warm-up mask: spk_t >= t_warmup, t_warmup ~ 1–2 s).
    # IMPLEMENTATION NOTE: 'El' and 'V0_span' are resolved from the params_NN NAMESPACE
    # (compile-time constants), NOT from the per-neuron state variable 'VT' (which is
    # assigned below and equals 0 at this point in the IC block).  Using V0_span avoids the
    # state-variable shadowing trap; see Brian2 namespace resolution order.
    N.V  = 'El + rand() * V0_span'   # uniform on [El, VT) at nominal VT = -51.0 mV
    N.gA = 0 * nS                    # adaptation deactivated (valid for swept g_A > 0 regime)

    # Default ICs for the per-neuron Brian2 parameters (so the library runs
    # standalone; shadowed per-neuron by run_args at sweep runtime). These MUST be
    # set: a name declared as a per-neuron parameter in eqs_NN resolves to the
    # state variable (default 0) before any namespace constant, so an unset gl /
    # gbarA / g_ampa would silently zero that term.
    N.sigma    = params_NN['sigma']
    N.gl       = params_NN['gl']
    N.gbarA    = params_NN['gbarA']
    N.delta_gA = params_NN['delta_gA']
    N.tauA     = params_NN['tauA']
    N.DeltaT   = params_NN['DeltaT']
    N.VT       = params_NN['VT']
    N.VA       = params_NN['VA']
    N.DeltaA   = params_NN['DeltaA']
    N.VR       = params_NN['VR']
    N.g_ampa   = params_NN['g_ampa']
    N.g_nmda   = params_NN['g_nmda']
    N.I_inj     = params_NN['I_inj']     # default per-neuron bias SCALE (shadowed by run_args)
    N.bias_unit = '(rand() - 0.5)'       # FROZEN unit pattern; standalone-runnable default
    
    
    
    
    # ----- SET POSITIONS AND CONNECTIONS -----
    # Position neurons on a grid

    Coordinates = get2D_rnd_coordinates(N.N,params_NN['c_min'],params_NN['c_max'],seed_neu)
    N.x = Coordinates[:,0]*um
    N.y = Coordinates[:,1]*um
    
    
    
    
    S = Synapses(N, N, model=eqs_Syn,
                 on_pre=pre,
                 on_post=post,
                 name='Synapse',
                 namespace={**params_Syn, 'Binomial_fun': Binomial_fun},
                 method='exponential_euler', dtype=float32,
                 )

    # -------------- Connections --------------
    # Errors are NOT swallowed here: a silent connection failure would corrupt
    # an entire HPC sweep without anyone noticing (Report 3 §A.2).
    if connections is True:
        # Random rule + distance-based bouton placement.
        if Syn_pdist is None:
            raise ValueError(
                "connections=True requires Syn_pdist (the dendritic-arbour PDF)."
            )
        S.connect(p=params_Syn['conn_prob'], condition='i != j')

        Syn_coordinates = get_synapse_coordinates(
            S, N,
            Syn_pdist['Syn_prob'].to_numpy(),
            Syn_pdist['Radius_val'].to_numpy(),
            seed_syn=seed_syn,
        )
        Syn_coordinates = np.vstack(Syn_coordinates)
        S.x_syn = Syn_coordinates[:, 0] * um
        S.y_syn = Syn_coordinates[:, 1] * um

    elif isinstance(connections, list):
        Source, Target = connections[0], connections[1]
        # Guard the zero-synapse case (e.g. conn_prob=0 for an isolated-neuron
        # search): Brian2's S.connect on empty index arrays raises on np.max of
        # a zero-size array. Skipping connect leaves S with no synapses, so the
        # summed r_ampa_tot/r_nmda_tot stay 0 and I_syn = 0 — exactly the
        # intended isolated-neuron limit.
        if len(Source) > 0:
            S.connect(i=Source, j=Target)

            # Restore bouton positions if provided (Report 3 §A.3).
            if syn_positions is not None:
                S.x_syn = np.asarray(syn_positions[:, 0]) * um
                S.y_syn = np.asarray(syn_positions[:, 1]) * um

    # ---- S Initialization ----
    # Skip entirely when S has no synapses (the conn_prob=0 isolated-neuron
    # case): writing a per-synapse variable before connect() raises in Brian2,
    # and there is nothing to initialise.
    if len(S) > 0:
        S.x_S = 1.0

        # Default ICs for per-synapse Brian2 parameters (so the library is runnable
        # standalone; shadowed by run_args sweep values at runtime). Must precede any
        # string IC that references Y_T (e.g. the ics=='rand' Y_S/Y_A inits below),
        # since Brian2 resolves the per-synapse state variable before the namespace const.
        S.EC50_ampa = params_Syn['EC50_ampa']
        S.EC50_nmda = params_Syn['EC50_nmda']
        S.n_ampa    = params_Syn['n_ampa']
        S.n_nmda    = params_Syn['n_nmda']
        S.x0        = params_Syn['x0']
        S.Y_T       = params_Syn['Y_T']
        S.O_G       = params_Syn['O_G']
        S.Omega_G   = params_Syn['Omega_G']

        # Random initialization of initial conditions — currently dormant.
        # Kept here so it can be re-enabled by passing ics='rand'.
        if ics == 'rand':
            S.usr = 'rand()'                                          # was: S.u_S
            S.x_S = 'rand()'
            S.Y_S = '1.2 * rho * Y_T * rand()'                        # was: rho_c (typo)
            S.Y_A = '1.2 * rho * Y_T * rand()'                        # slow spillover cleft var

    return N, S
        
    



# -------------- ASTROCYTE GROUP --------------
def _astrocyte_steady_state(params_astroGT, t_max=300.0):
    """
    Integrate the *isolated* astrocyte ODEs (no synaptic input, no GJ coupling)
    to a quiescent fixed point.

    Returns a dict with values in dimensionless units (Gamma_A, h) or in
    Brian2 Quantity (I, C in mole). Used to seed Astro.Gamma_A, Astro.I,
    Astro.C, Astro.h with biologically meaningful initial conditions instead
    of Brian2's default zeros (Report 3 §A.1).

    Parameters
    ----------
    params_astroGT : dict
        Output of `get_Astroparam()`. Units handled internally — the function
        strips units, integrates with scipy.odeint in plain numerics
        (concentrations in umol, time in seconds), and re-attaches units to
        the returned C and I.
    t_max : float
        Integration horizon in seconds. 300 s is several times the slowest
        time constant (tau_h ~ 35 s at low C) and reaches the fixed point.

    Returns
    -------
    dict {'Gamma_A': float, 'I': mole, 'C': mole, 'h': float}
    """
    p = params_astroGT

    # Strip units once.
    p_num = {
        'O_beta':   p['O_beta']   / (umole/second),
        'O_delta':  p['O_delta']  / (umole/second),
        'K_delta':  p['K_delta']  / umole,
        'O_3K':     p['O_3K']     / (umole/second),
        'K_D':      p['K_D']      / umole,
        'K_3K':     p['K_3K']     / umole,
        'Omega_5P': p['Omega_5P'] * second,
        'Omega_C':  p['Omega_C']  * second,
        'Omega_L':  p['Omega_L']  * second,
        'O_P':      p['O_P']      / (umole/second),
        'K_P':      p['K_P']      / umole,
        'C_T':      p['C_T']      / umole,
        'rho_A':    p['rho_A'],
        'd_1':      p['d_1']      / umole,
        'd_2':      p['d_2']      / umole,
        'd_3':      p['d_3']      / umole,
        'd_5':      p['d_5']      / umole,
        'O_2':      p['O_2']      * umole * second,    # 1/(umol*s) -> dimensionless after multiplying by umol*s
    }

    def rhs(state, t):
        Gamma_A, I, C, h = state
        # With Y_bias=0 and Y_extra=0, the Gamma_A source vanishes.
        # Gamma_A then decays exponentially; starting at 0 it stays at 0.
        m_inf = (I / (I + p_num['d_1'])) * (C / (C + p_num['d_5']))
        Q_2   = p_num['d_2'] * (I + p_num['d_1']) / (I + p_num['d_3'])
        h_inf = Q_2 / (Q_2 + C)
        tau_h = 1.0 / (p_num['O_2'] * (Q_2 + C))

        dGamma_A = 0.0
        # NOTE: I_exogenous is intentionally OMITTED here (notes §4). This routine
        # computes the isolated-cell quiescent IC; the live dI/dt includes
        # + I_exogenous, so for I_bias far from this IC the live model exhibits a
        # startup transient toward I_bias. This is accepted behavior — do NOT add
        # I_exogenous to this integrator (it would pull the IC and defeat its purpose).
        dI = (p_num['O_delta'] / (1 + I / p_num['K_delta'])) * \
             (C**2 / (C**2 + p_num['K_delta']**2)) \
             - p_num['O_3K'] * (C**4 / (C**4 + p_num['K_D']**4)) * \
               (I / (I + p_num['K_3K'])) \
             - p_num['Omega_5P'] * I
        dC = (p_num['Omega_C'] * m_inf**3 * h**3 + p_num['Omega_L']) * \
             (p_num['C_T'] - (1 + p_num['rho_A']) * C) \
             - p_num['O_P'] * (C**2 / (C**2 + p_num['K_P']**2))
        dh = (h_inf - h) / tau_h
        return [dGamma_A, dI, dC, dh]

    # Initial guess: low IP3, low Ca, h near 1.
    y0 = [0.0, 0.1, 0.05, 0.9]
    t = np.linspace(0.0, t_max, 2001)
    sol = odeint(rhs, y0, t, rtol=1e-8, atol=1e-10, mxstep=5000)

    return {
        'Gamma_A': float(sol[-1, 0]),
        'I':       float(sol[-1, 1]) * umole,
        'C':       float(sol[-1, 2]) * umole,
        'h':       float(sol[-1, 3]),
    }


def astrocyte_connections(Astrocyte_group, Connection_dist,
                          topology_mode='distance',
                          Neuron_group=None,
                          gj_max_dist=150.0):
    """
    Build the gap-junction adjacency between astrocytes.

    Two modes are supported, selected by `topology_mode`:

      * 'distance' (legacy, default for backward compatibility):
        Every pair of astrocytes whose centre-to-centre distance is less than
        `Connection_dist` (in µm) is GJ-coupled bidirectionally.

      * 'wallach':
        Two astrocytes are GJ-coupled iff (i) they share a border in the
        JOINT Voronoi tessellation of neurons ∪ astrocytes (i.e. they are
        Delaunay-adjacent in the joint point set, Wallach et al. 2014) AND
        (ii) their centre-to-centre distance does not exceed `gj_max_dist`.
        Requires `Neuron_group` to expose `.x` and `.y` (positions in metres
        or as Brian2 Quantity).

    Both modes return BIDIRECTIONAL pair lists: each undirected edge appears
    as both (i, j) and (j, i). Self-loops are excluded.

    Parameters
    ----------
    Astrocyte_group : Brian2 NeuronGroup
        Must expose `.x_astro` and `.y_astro` (Quantity in metres).
    Connection_dist : float
        Radius in µm for the 'distance' mode (ignored in 'wallach' mode).
    topology_mode : {'distance', 'wallach'}
    Neuron_group : Brian2 NeuronGroup, optional
        Required for 'wallach' mode. Must expose `.x` and `.y`.
    gj_max_dist : float
        Soft cap in µm applied on top of the Voronoi adjacency in 'wallach'
        mode (ignored in 'distance' mode). Default 150 µm.

    Returns
    -------
    Source, Target : 1-D arrays of int
        Bidirectional pair list.
    """
    # Astrocyte coordinates as plain NumPy in µm.
    x_pos = np.array(Astrocyte_group[:].x_astro / um)
    y_pos = np.array(Astrocyte_group[:].y_astro / um)
    A_pos = np.column_stack((x_pos, y_pos))

    if topology_mode == 'wallach':
        if Neuron_group is None:
            raise ValueError(
                "astrocyte_connections(topology_mode='wallach', ...) "
                "requires Neuron_group so that the joint (neurons ∪ "
                "astrocytes) Voronoi tessellation can be computed."
            )
        N_pos = np.column_stack((
            np.asarray(Neuron_group[:].x / um),
            np.asarray(Neuron_group[:].y / um),
        ))
        return voronoi_astro_gj_pairs(N_pos, A_pos, gj_max_dist=gj_max_dist)

    elif topology_mode == 'distance':
        Astro_positions = KDTree(A_pos)

        Source = []
        Target = []

        for astro_idx, astro in enumerate(A_pos):
            A_idx = Astro_positions.query_radius(astro.reshape(1, -1),
                                                 r=Connection_dist)
            A_idx = np.array(A_idx[0])

            # Drop self (KDTree includes the query point itself).
            A_idx = A_idx[A_idx != astro_idx]

            if A_idx.size == 0:
                continue

            source_astro = np.full(len(A_idx), astro_idx)
            Source.append(source_astro)
            Target.append(A_idx)

        if not Source:
            # No astrocyte has any neighbour — return empty arrays.
            return np.array([], dtype=int), np.array([], dtype=int)

        return np.concatenate(Source), np.concatenate(Target)

    else:
        raise ValueError(
            f"astrocyte_connections: unknown topology_mode={topology_mode!r}. "
            "Expected 'wallach' or 'distance'."
        )


def Astrocyte_Group(N_astro, Simulated_network, seed_astro=None, ics='steady',
                    connections=None,
                    topology_mode='distance',
                    Neuron_group=None,
                    gj_max_dist=150.0):
    """
    Build the astrocyte NeuronGroup ('Astrocyte') and the gap-junction Synapses
    group ('Gap_junctions'). In `'Astrocytic'` mode additionally builds a
    PoissonGroup with synaptic glutamate input.

    Parameters
    ----------
    N_astro : int
    Simulated_network : {'Full', 'Neuronal', 'Astrocytic'}
    seed_astro : int or None
        Used both for astrocyte placement and for the per-cell jitter on the
        steady-state initial conditions.
    ics : {'steady', 'rand', None}
        - 'steady' (default): integrate the isolated astrocyte ODEs to the
          quiescent fixed point, then jitter each astrocyte by ±5%
          (Report 3 §A.1). This is essential for HPC runs <~ 60 s where
          starting from zero leaves the IP3R inactivation gate untrained.
        - 'rand': legacy behaviour, fully random uniform initial conditions.
        - None: leave Brian2 defaults (all zeros) — not recommended.
    connections : {None, True, list}
        See Neuronal_Network docstring.
    topology_mode : {'distance', 'wallach'}
        Only consulted when `connections is True`. Selects the rule used to
        generate the GJC adjacency from cell positions:
          - 'distance' (legacy default): pairs within `conn_dist` µm.
          - 'wallach': Delaunay-adjacency in the joint (neurons ∪ astrocytes)
            tessellation, capped at `gj_max_dist` µm.
        Ignored when `connections` is a list (explicit pair list — pre-built
        upstream by e.g. HPC_single_run.build_topology).
    Neuron_group : Brian2 NeuronGroup, optional
        Required when `topology_mode='wallach'` and `connections is True`.
    gj_max_dist : float
        Soft cap in µm for the Wallach branch (ignored in 'distance' mode).

    Returns
    -------
    Astro, GJ                     for 'Full' / 'Neuronal'
    Astro, GJ, P, Glu_Input       for 'Astrocytic'
    """

    # ------ Astrocyte core equations ------

    eqs_A = Equations('''
        # Fraction of activated astrocyte receptors:
        dGamma_A/dt = O_N * (Y_bias+Y_extra*spill_over)**n * (1 - Gamma_A) -
                      Omega_N*(1 + zeta * C/(C + K_KC)) * Gamma_A : 1 
    
        # IP_3 dynamics (de Pittà form + exogenous tonic drive toward I_bias):
        dI/dt = O_beta * Gamma_A + O_delta/(1 + I/K_delta) * C**2/(C**2 + K_delta**2) -
                O_3K * C**4/(C**4 + K_D**4) * I/(I + K_3K) - Omega_5P*I +
                I_coupling_tot + I_exogenous : mole 
    
        # Exogenous tonic drive toward I_bias (de Pittà soft set-point, threshold-gated):
        delta_I_bias = I - I_bias : mole
        I_exogenous  = -F/2 * (1 + tanh((abs(delta_I_bias) - I_Theta)/omega_I)) * sign(delta_I_bias) : mole/second
    
        # diffusion between astrocytes:
        I_coupling_tot : mole/second
       
    
        # Ca^2+-induced Ca^2+ release:
        dC/dt = (Omega_C * m_inf**3 * h**3 + Omega_L) * (C_T - (1 + rho_A)*C) -
                O_P * C**2/(C**2 + K_P**2) : mole 
        dh/dt = (h_inf - h)/tau_h : 1  
        m_inf = I/(I + d_1) * C/(C + d_5) : 1
        h_inf = Q_2/(Q_2 + C) : 1
        tau_h = 1/(O_2 * (Q_2 + C)) : second
        Q_2 = d_2 * (I + d_1)/(I + d_3) : mole
    
        # External neurotransmitter stimulation
        Y_bias : mole
        # Neurotransmitter concentration in the extracellular space
        Y_extra : mole
    
        # Additional (optional) coordinates (for spatial network implementation)
        x_astro : meter
        y_astro : meter

        # Per-astrocyte Brian2 parameters (shadow namespace constants at runtime).
        # F, I_Theta, omega_I are the single source of truth — the Gap_junctions
        # group reads them via the _post suffix (see Gap_Eq below).
        O_beta   : mole/second
        O_3K     : mole/second
        Omega_5P : 1/second
        I_bias   : mole
        F        : mole/second
        I_Theta  : mole
        omega_I  : mole
        ''')
       
       
       
    Params_astroGT = get_Astroparam()

    # The definition of a threshold and reset mechanism in the astrocyte group
    # allows to use SpikeMonitors to estimate the frequency of oscillations
    Astro = NeuronGroup(N_astro, eqs_A,
                        threshold='C>C_osc',
                        refractory='C>C_osc',
                        method='rk4',
                        namespace=Params_astroGT,
                        name='Astrocyte', dtype=float32)

    # Default ICs for per-astrocyte Brian2 parameters (so the library is runnable
    # standalone; shadowed by run_args sweep values at runtime). Required because
    # these names are now state variables in eqs_A: Brian2 resolves the state var
    # (default 0) before the namespace constant, so an unset O_beta/Omega_5P/etc.
    # would zero out the corresponding dynamics term. F/I_Theta/omega_I here are the
    # single source of truth also read by the Gap_junctions group via the _post suffix.
    Astro.O_beta   = Params_astroGT['O_beta']
    Astro.O_3K     = Params_astroGT['O_3K']
    Astro.Omega_5P = Params_astroGT['Omega_5P']
    Astro.I_bias   = Params_astroGT['I_bias']
    Astro.F        = Params_astroGT['F']
    Astro.I_Theta  = Params_astroGT['I_Theta']
    Astro.omega_I  = Params_astroGT['omega_I']

    # ----- Initial conditions (Report 3 §A.1) -----
    if ics == 'rand':
        # Legacy random uniform ICs.
        Astro.Gamma_A = 'rand()'
        Astro.I = '3*rand()*umole'
        Astro.C = '1.5*rand()*umole'
        Astro.h = 'rand()'

    elif ics == 'steady':
        # Steady-state from integrating the isolated astrocyte equations,
        # plus ±5% per-cell jitter to break perfect symmetry.
        ss = _astrocyte_steady_state(Params_astroGT)
        rng_ic = np.random.default_rng(seed_astro)
        jitter = lambda: 1.0 + 0.05 * (2.0 * rng_ic.random(N_astro) - 1.0)

        Astro.Gamma_A = ss['Gamma_A']                          # ~0; jitter would be meaningless
        Astro.I       = (ss['I'] / umole) * jitter() * umole
        Astro.C       = (ss['C'] / umole) * jitter() * umole
        Astro.h       = ss['h'] * jitter()
        print(f"  Astrocyte ICs from steady state: "
              f"Gamma_A={ss['Gamma_A']:.4f}, "
              f"I={ss['I']/umole:.4f} uM, "
              f"C={ss['C']/umole:.4f} uM, "
              f"h={ss['h']:.4f} (±5% per-cell jitter applied).")

    # If ics is None / False, leave Brian2's default zeros — not recommended.

    # ----- SET POSITIONS -----
    Coordinates = get2D_rnd_coordinates(N_astro, Params_astroGT['c_min'], Params_astroGT['c_max'], seed_astro)
    Astro.x_astro = Coordinates[:, 0] * um
    Astro.y_astro = Coordinates[:, 1] * um


    # ----- Gap-junction based astro links -----

    Gap_Eq = Equations('''
        delta_I = I_post - I_pre : mole
        I_coupling = -F_post/2*(1 + tanh((abs(delta_I) - I_Theta_post)/omega_I_post))*sign(delta_I) : mole/second
        I_coupling_tot_post = I_coupling : mole/second (summed)
    ''')

    GJ = Synapses(Astro, Astro,
                  model=Gap_Eq,
                  method='rk4',
                  namespace=Params_astroGT,
                  name='Gap_junctions', dtype=float32,
                  )

    # ----- Connections -----
    # Two rules are supported (selected by `topology_mode`):
    #   - 'distance' (legacy): KDTree-based pairs within `conn_dist`.
    #     Reference: "A Computational Model of Interactions Between Neuronal
    #     and Astrocytic Networks..." (legacy approach used in early reports).
    #   - 'wallach' : Delaunay-adjacency in the joint Voronoi tessellation of
    #     neurons ∪ astrocytes, capped at `gj_max_dist` µm.
    #     Reference: Wallach et al. 2014, PLOS Comput. Biol. 10(12) e1003964;
    #     and De Pittà & Berry (eds.), Computational Glioscience, 2019, ch. 7.
    if connections is True:
        Source, Target = astrocyte_connections(
            Astro, Params_astroGT['conn_dist'],
            topology_mode=topology_mode,
            Neuron_group=Neuron_group,
            gj_max_dist=gj_max_dist,
        )
        GJ.connect(i=Source, j=Target)

    elif isinstance(connections, list):
        Source, Target = connections[0], connections[1]
        GJ.connect(i=Source, j=Target)

    if Simulated_network == 'Astrocytic':
        import random
        random.seed(seed_astro)                                # was: random.seed(sed) — undefined

        # ---------- EXTERNAL STIMULATION ----------
        Params_astroGT.update({'tau_glustim': 25*ms})          # As for synapse params
        Params_astroGT.update({'Y_bias_max': 1*mmole})         # Maximum glutamate concentration
        Params_astroGT.update({'poisson_rate': 2*Hz})          # Rate of the Poisson process
        Params_astroGT.update({'N_stim': 40})                  # Number of stimulated astros
        
        
        # --- Poisson process ---
        P = PoissonGroup(Params_astroGT['N_stim'], Params_astroGT['poisson_rate'])
        
        
        # --- Equations ---
        Glu_stim_Eq = Equations('''
                                
                                dY_bias_in/dt = -Y_bias_in/tau_glustim : mole (clock-driven)
                                Y_bias_post = Y_bias_in : mole (summed)
                                
                                ''') 
        pre = '''
        
            Y_bias_in += Y_bias_max  
        
        
            '''
            
        post = None
        
        
        
        Glu_Input = Synapses(P, Astro, model=Glu_stim_Eq,
                       on_pre=pre, on_post=post,namespace=Params_astroGT,method='exponential_euler')
    
        # Randomly choose N_stim neurons
        random_astro = random.sample(range(0, Astro.N), Params_astroGT['N_stim'])
        
        Glu_Input.connect(i=np.arange(Params_astroGT['N_stim']), j=random_astro)
    
    
        
        
        
        
        
        return Astro, GJ,P,Glu_Input
    
    else:
        return Astro, GJ
    
    
    
# ------ Gliotransmission ------
def Gliotransmission(N_astro, Astro, ics='jitter', seed_astro=None):
    """
    Build the gliotransmitter-release NeuronGroup ('Gliot_release').

    Its `C` variable is linked to Astro.C via `linked_var`, so threshold
    crossings of C trigger gliotransmitter release.

    Parameters
    ----------
    ics : {'jitter', 'rand', None}
        - 'jitter' (default): x_A initialised to 1 with ±5% per-cell jitter
          (Report 3 §A.1, slight symmetry break); G_A = 0.
        - 'rand': legacy random ICs (currently dormant).
        - None: x_A = 1 exactly, G_A = 0.
    seed_astro : int or None
        Seed for the ±5% jitter on x_A.
    """
    eqs_GT = Equations('''
        # Gliotransmitter
        C : mole (linked)
        dx_A/dt = Omega_A * (1 - x_A) : 1     # Fraction of gliotransmitter resources available for release
        dG_A/dt = -Omega_e*G_A : mole         # Gliotransmitter concentration in the extracellular space

        # Per-unit Brian2 parameters (shadow namespace constants at runtime).
        # C_Theta is also the spike threshold/refractory condition below.
        C_Theta : mole
        U_A     : 1
        G_T     : mole
    ''')
    gliot_release = '''
        G_A += rho_e * G_T * U_A * x_A
        x_A -= U_A * x_A
    '''

    Params_astroGT = get_Astroparam()
    Glio_release = NeuronGroup(N_astro, eqs_GT,
                               threshold='C>C_Theta',
                               refractory='C>C_Theta',
                               reset=gliot_release,
                               method='rk4',
                               name='Gliot_release',
                               namespace=Params_astroGT, dtype=float32)

    # Default ICs
    Glio_release.G_A = 0.0 * mole
    Glio_release.C = linked_var(Astro, 'C')

    # Default ICs for per-unit Brian2 parameters (so the library is runnable
    # standalone; shadowed by run_args sweep values at runtime). Must precede any
    # string IC referencing G_T (e.g. the ics=='rand' branch below).
    Glio_release.C_Theta = Params_astroGT['C_Theta']
    Glio_release.U_A     = Params_astroGT['U_A']
    Glio_release.G_T     = Params_astroGT['G_T']

    if ics == 'jitter':
        rng_ic = np.random.default_rng(
            None if seed_astro is None else seed_astro + 1
        )
        # ±5% around the full reserve x_A = 1
        Glio_release.x_A = 1.0 - 0.05 * rng_ic.random(N_astro)
    elif ics == 'rand':
        # Legacy dormant branch — fixed (was: 'synapses' undefined). Report 3 §B.2.
        Glio_release.x_A = 'rand()'
        Glio_release.G_A = '1.2 * rho_e * G_T * rand()'
    else:
        Glio_release.x_A = 1.0

    return Glio_release


# -------------- SYNAPSE-ASTRO LINK ---------------

def Synapse_to_astro(synapse, Astro, connections,
                     topology_mode='distance',
                     stoa_cutoff=70.0):
    """
    Build the (summed) glutamate-spillover link from synapses onto astrocytes.

    Two `topology_mode` rules are supported when `connections is True`:

      * 'distance' (legacy):
        Brian2-internal Gaussian-probability connection within a hard cutoff.
        Each synapse-astrocyte pair within `Conn_syn_astro_cutoff` µm is
        sampled independently with p = exp(-d² / (2·sigma_A²)). As a
        consequence a single synapse can end up linked to multiple astrocytes.

      * 'wallach' (new default, Wallach et al. 2014-compatible):
        Each synaptic bouton is assigned to AT MOST ONE astrocyte — the
        nearest one within `stoa_cutoff` µm. Distance is measured from the
        bouton position (x_syn, y_syn).

    Parameters
    ----------
    connections : {True, list}
        - True: build connectivity from cell positions using `topology_mode`.
          Requires `synapse` to expose `.x_syn`, `.y_syn` and `Astro` to
          expose `.x_astro`, `.y_astro`.
        - list [Source, Target]: explicit (i_synapse, j_astrocyte) pairs
          (typically supplied by HPC_single_run.build_topology).
    topology_mode : {'distance', 'wallach'}
        Only consulted when `connections is True`.
    stoa_cutoff : float
        Hard distance cutoff in µm for the 'wallach' branch. Default 70 µm.

    Returns
    -------
    Syn_Astro : Synapses
    Connections_list : [Target, Source]
        Swapped pairs ready to be passed to `Astro_to_Syn` (which needs
        (i_astrocyte, j_synapse)).
    """
    Syn_Astro = Synapses(synapse, Astro,
                         model='''
                         # neurotransmitter concentration in the extracellular space
                         Y_extra_post = Y_A_pre : mole (summed)
                         ''',
                         namespace=synapse.namespace,
                         method='rk4',
                         name='ecs_syn_to_astro')

    # Errors NOT swallowed (Report 3 §A.2).
    if connections is True:
        if topology_mode == 'wallach':
            # Pull bouton + astrocyte positions out into numpy and run the
            # 1-to-1 nearest-astrocyte assignment.
            bouton_pos = np.column_stack((
                np.asarray(synapse.x_syn / um),
                np.asarray(synapse.y_syn / um),
            ))
            A_pos = np.column_stack((
                np.asarray(Astro.x_astro / um),
                np.asarray(Astro.y_astro / um),
            ))
            Source, Target = nearest_astro_for_synapse(
                bouton_pos, A_pos, stoa_cutoff=stoa_cutoff
            )
            Syn_Astro.connect(i=Source, j=Target)
            Connections_list = [Target, Source]

        elif topology_mode == 'distance':
            p_conn = ('exp(- ((sqrt((x_syn_pre - x_astro_post)**2 + '
                      '(y_syn_pre - y_astro_post)**2))**2) / (2 * sigma_A**2))')
            Syn_Astro.connect(
                condition='sqrt((x_syn_pre - x_astro_post)**2 + (y_syn_pre - y_astro_post)**2) < Conn_syn_astro_cutoff',
                p=p_conn,
            )
            Source = list(Syn_Astro.i)         # Synapse as pre unit
            Target = list(Syn_Astro.j)         # Astro as target unit
            Connections_list = [Target, Source]

        else:
            raise ValueError(
                f"Synapse_to_astro: unknown topology_mode={topology_mode!r}. "
                "Expected 'wallach' or 'distance'."
            )

    elif isinstance(connections, list):
        Source, Target = connections[0], connections[1]
        Syn_Astro.connect(i=Source, j=Target)
        Connections_list = [Target, Source]

    else:
        raise ValueError(
            "Synapse_to_astro: `connections` must be True or a list "
            "[Source, Target]."
        )

    return Syn_Astro, Connections_list


def Astro_to_Syn(Glio_release, synapse, connections):
    """
    Build the (summed) gliotransmitter-spillover link from gliotransmitter
    release units onto synapses.

    `connections` is the Connections_list returned by Synapse_to_astro
    (already swapped: [Astro_source, Synapse_target]).
    """
    Astro_Syn = Synapses(Glio_release, synapse,
                         model='''
                         # gliotransmitter concentration in the extracellular space
                         G_A_syn_post = G_A_pre : mole (summed)
                         ''',
                         method='rk4',                     # was: 'gsl ' (Report 3 §B.4)
                         name='ecs_astro_to_syn', dtype=float32,
                         )

    Source, Target = connections[0], connections[1]
    Astro_Syn.connect(i=Source, j=Target)

    return Astro_Syn


# --------------- ELECTRODE RECORDINGS ---------------




def generate_grid_points(n_rows, n_cols, pitch,x0,y0):
    """
    Generates the 2D coordinates for a grid of electrodes.

    Args:
        n_rows (int): Number of rows in the electrode grid.
        n_cols (int): Number of columns in the electrode grid.
        pitch_micrometers (float): The distance between adjacent electrodes
                                   in micrometers.
        x0,y0 = bottom left end coordinates.
    Returns:
        numpy.ndarray: A 2D array where each row represents the (x, y)
                       coordinates of an electrode point, in micrometers.
    """
    points = []
    # Pitch is used directly in micrometers as requested
    

    for r in range(n_rows):
        for c in range(n_cols):
            x = x0 + c * pitch
            y = y0 + r * pitch
            points.append((x, y))

    return np.array(points)



def Electrode_recording(MEA_dict,Neuron_group,State_Monitor,electrode_dist,neuron_radius,electrode_radius):
    
    # Generate the KDTree form the neuronal position data
    x_pos = np.array(Neuron_group[:].x/um)
    y_pos = np.array(Neuron_group[:].y/um)
    
    pos = np.column_stack((x_pos, y_pos))

    # Generate the KDTree
    Neuron_positions = KDTree(pos)
    
    
    Electrode_recordings = {}
    
    for key in MEA_dict.keys():
        
        rec_sites = MEA_dict[key]
        
        Electrode_rec = Electrode_trace(rec_sites,Neuron_group,Neuron_positions,State_Monitor,electrode_dist,neuron_radius,electrode_radius) 

        Electrode_recordings[key] = Electrode_rec
        
        
    return Electrode_recordings
    
    
    

# def Electrode_traces(rec_sites, Neuron_group, Neuron_positions, State_Monitor, electrode_dist, neuron_radius, electrode_radius):
#     """
#     Optimized version of the Electrode_trace function using vectorization.
    
    
   
#         For neurons below d_lim a simplyfied EEI model is used
#         For neurons below d_lim a Dipole approximation is implemented 
        
#         Two contributes for background noise: 1) distant neurons, 2) White noise
        
#         White noise: gaussian distribution of mean 0 and std 1uV --> 1e-3 mV
        
#         For each recording site the sum of all the contributing neurons is taken and 
#         for the whole electrode the mean across the recording sites.
        
#         electrode_dist = max senstivity distance of the electrode
#         Neuron_psoitions = sklearn.neighbors.NearestNeighbors object
#         neuron_radius = radiu of the neurons expressed in micrometers.
#                 it is used to because the distance is calculated between centers
#         electrode_radius = radius of the electrode  
#         # --- Papers:
            
#             1) Multi-program approach for simulating recorded extracellular signals
#               generated by neurons coupled to microelectrode arrays.
              
#             2) A Detailed and Fast Model of Extracellular Recordings.
        
        
#     """
#     d_lim = 50  # [um]
#     White_noise = 1e-3  # [mV]
#     Rho_s = 0.7 * 1e6  # [ Ohm * um ] Saline bath resistivity
    
#     # Pre-allocate a list to store the voltage traces for each site
#     site_voltage_traces = []

#     dt_ = defaultclock.dt
    
#     # Pre-calculate the neuron state monitor data for efficiency
#     # This assumes State_Monitor is a list-like object
#     state_V_mV = [monitor.V/mV for monitor in State_Monitor]
#     state_I_mA = [monitor.I_cell/mA for monitor in State_Monitor]
    
#     for site in rec_sites:
#         # For each recording site extract the recorded neurons
#         NN_idx, NN_dist = Neuron_positions.query_radius(site.reshape(1, -1), r=electrode_dist, return_distance=True)
#         NN_idx = NN_idx[0]
#         NN_dist = NN_dist[0]
        
#         if len(NN_idx) == 0:
#             site_voltage_traces.append(np.zeros_like(state_V_mV[0]))
#             continue
        
#         # Determine which model to use with boolean indexing
#         close_neurons_mask = NN_dist < (d_lim + neuron_radius + electrode_radius)
        
#         # Initialize an array for voltages for the current site
#         num_time_steps = state_V_mV[NN_idx[0]].shape[0]
#         voltages_for_site = np.zeros(num_time_steps)
        
#         # Handle distant neurons (Dipole approximation)
#         distant_idx = NN_idx[~close_neurons_mask]
#         if len(distant_idx) > 0:
#             # Get pre-calculated V and I data for distant neurons
#             distant_V = np.array([state_V_mV[i] for i in distant_idx])
#             distant_dist = NN_dist[~close_neurons_mask]
            
#             # Vectorized calculation for distant neurons
#             V_dipole = distant_V * (1 / (distant_dist**2))[:, np.newaxis]
#             voltages_for_site += np.sum(V_dipole, axis=0)

#         # Handle close neurons (Monopole model)
#         close_idx = NN_idx[close_neurons_mask]
#         if len(close_idx) > 0:
#             # Get pre-calculated V and I data for close neurons
#             close_I = np.array([state_I_mA[i] for i in close_idx])
#             close_dist = NN_dist[close_neurons_mask]
            
#             # Vectorized calculation for close neurons
#             V_monopole = (Rho_s * close_I) / (4 * np.pi * close_dist[:, np.newaxis])
#             voltages_for_site += np.sum(V_monopole, axis=0)
        
#         # Save the summed voltage trace for the current site
#         site_voltage_traces.append(voltages_for_site)

#     # --- MEAN ---
#     # Take the mean across all recording sites
#     if len(site_voltage_traces) > 0:
#         rec_list_ = np.vstack(site_voltage_traces)
#         Electrode_trace = np.mean(rec_list_, axis=0)
#     else:
#         # Handle the case where no neurons were found near any site
#         return np.zeros(Neuron_group.N) # Adjust size as needed

#     # Add white noise
#     white_noise_vector = np.random.normal(loc=0, scale=White_noise, size=len(Electrode_trace))
#     Electrode_trace = Electrode_trace + white_noise_vector
    
#     return Electrode_trace    
    
    
    



def Electrode_trace(rec_sites,Neuron_group,Neuron_positions,State_Monitor,electrode_dist,neuron_radius,electrode_radius):

    '''
    For neurons below d_lim a simplyfied EEI model is used
    For neurons below d_lim a Dipole approximation is implemented 
    
    Two contributes for background noise: 1) distant neurons, 2) White noise
    
    White noise: gaussian distribution of mean 0 and std 1uV --> 1e-3 mV
    
    For each recording site the sum of all the contributing neurons is taken and 
    for the whole electrode the mean across the recording sites.
    
    electrode_dist = max senstivity distance of the electrode
    Neuron_psoitions = sklearn.neighbors.NearestNeighbors object
    neuron_radius = radiu of the neurons expressed in micrometers.
            it is used to because the distance is calculated between centers
    electrode_radius = radius of the electrode  
    # --- Papers:
        
        1) Multi-program approach for simulating recorded extracellular signals
          generated by neurons coupled to microelectrode arrays.
          
        2) A Detailed and Fast Model of Extracellular Recordings.
    
    '''
    
    # For each recording site extract the recorded neurons
    d_lim = 50 # [um]
    White_noise = 1e-3 # [mV]
    Rho_s = 0.7 * 1e6 #[ Ohm * um ] Saline bath resistivity 
    Site_voltages = {}
    s = 0
    dt_ = defaultclock.dt
    for site in rec_sites:
        # For each recording site extract the recorded neurons
        NN_idx,NN_dist = Neuron_positions.query_radius(site.reshape(1, -1), r=electrode_dist, return_distance=True)
        NN_idx = NN_idx[0]
        NN_dist = NN_dist[0]
        
        
        Voltages = []
        for neu in range(len(NN_idx)):
           
           
        # First evaluate which model to use
        
            if NN_dist[neu] >= d_lim+neuron_radius+electrode_radius:
                
                # continue
                
                V  = State_Monitor[NN_idx[neu]].V/mV * 1/(NN_dist[neu]**2)
            
                Voltages.append(V)
                
            else:
                
                

                
                # Monopole
                V = (Rho_s*State_Monitor[NN_idx[neu]].I_cell/mA)/(4*np.pi*NN_dist[neu])
                # V = Voltage_trace(alpha_,beta_, State_Monitor[NN_idx[neu]].V/mV* 1/(NN_dist[neu]**2) ,dt_)
                
                # Evaluate the voltage seen by the electrode
               
                
                Voltages.append(V)
                
                
                
                
                
                
                
            
        # Sum column-wise
        
        Voltages_sum = np.sum(np.array(Voltages),axis=0)
        
        # Save
        Site_voltages[s] = Voltages_sum
        s = s+1
                                          

    # --- MEAN ---
    
    # Extract traces
    rec_list = [Site_voltages[key] for key in Site_voltages.keys()]   
    rec_list_ = np.vstack(rec_list)
    
    # Take the mean
    Electrode_trace_ = np.mean(rec_list_, axis=0)
    
    # Add white noise
    white_noise_vector = np.random.normal(loc=0, scale=White_noise, size=len(Electrode_trace_))

    Electrode_trace_ = Electrode_trace_ + white_noise_vector
    return Electrode_trace_
                 

def Voltage_trace(alpha,beta,V,dt_):
    
    '''
    V is in millivolt
    
    returns the Convolved trace. SHould be in [mV]
    
    alpha and beta parameters are defined so that the integration constant is 
    expressed in ms.
    
    '''
    dt_ = dt_/ms
    
    V_dot = np.zeros(len(V))
    
    # First define the derivative vector of the intracellular membrane voltage
    # The resulting vector is always one sample less than the original
    V_dot[1:] = np.diff(V)
    
    
    # Integrate with a simple first-order Euler
    V_convolved = np.zeros(len(V))
    
     
    for t in np.arange(1,len(V)):
        
        V_convolved[t] = V_convolved[t-1] + (-alpha*V_convolved[t-1]  - beta*V_dot[t])*(dt_)
    
    
    return V_convolved
    
    
    
    
    
    
    

                                     

def TF_params(d):

    '''
    Evaluates the transfer function parameters in base of the neuronal proximity
    
    d = distance in micrometers
    
    1 ms = 1 MOhm * 1 nF
    '''  
    
    # C_e = 1.14 * 1e-9 # [F]
    # C_hd = 17.45 * 1e-12 # [F]
    # R_e = 0.14 * 1e6 # [Ohm]
    # eps_IHP = 6
    # eps_OHP = 32
    # eps_0  = 8.85*1e-12 # [F/m]
    # d_IHP = 0.3 *1e-9 # [m]
    # d_OHP = 0.7 *1e-9 # [m]
    # eps_D = 50 
    # N = 6.022 *1e23 # [1/mol]
    # q = 1.6021 *1e-19 # [C]
    # n_0 = 150 * 1e-3 #[mol]
    # k= 1.38064 *1e-23 # [J/K]
    # T= 300 # [K]
    
    
    
    # C_e = 1.14 # [nF]
    # C_hd = 17.45 * 1e-3 # [nF]
    # R_e = 0.14 # [MOhm]
    
    # rho_s = 0.7*1e-6 # [MOhm * m] Saline bath resistance
    
    # Area_ratio = 0.5 # Approx the electrode is twice the somata.
    
    # d_ = 70 * 1e-9 #[m]
    
    # R_seal = (rho_s/d_) * Area_ratio
    
    # C_h1 = (eps_0*eps_IHP*Area)/(d_IHP)
    # C_h2 = (eps_0*eps_OHP*Area)/(d_OHP-d_IHP)
    # C_d = (q*np.sqrt(2*eps_0*eps_D*k*T*n_0*N)*Area)/(k*T)
    
    # C_hd_inv = (1/C_h1) + (1/C_h2) + (1/C_d)
    # C_hd = 1/C_hd_inv
    
    
    
    
    
    alpha = (R_e + R_seal) / ( (R_e*R_seal)  *  (C_hd + C_e) )
    beta = C_hd/(C_hd + C_e)


    return alpha,beta       



def Get_12grid(pitch):
    
    '''
    Generates a set of coordinates for the 12 electrode MEA configuration
    
      x x
    x x x x
    x x x x
      x x
 
    '''
    
    shift = 100 # [um]
    x0 = 0 + shift
    y0 = 0 + shift
    pitch = 300 # [um]
    
    Grid_raw = generate_grid_points(4, 4, pitch,x0,y0)
    
    
    # The configuration is 12 electrodes. For this reason we will
    # discard the following points: 0,3,12,15
    
    Idx_remove = np.array([0,3,12,15])
    
    Grid = np.delete(Grid_raw, Idx_remove, axis=0)
    
    return Grid
    




    
def Recording_sites(pitch_recsites,shift,Grid):
    MEA_dict = {}
    
    el = 0
    for point in Grid:  
        
        # The x0 and y0 are the bottom left coordinates of the first rec site.
        # the 'point' coordinate is the center. A shift in coordinates is needed.
        # The 'point' coordinates are shifted along the diagonal about half the diameter.
        # Both x and y of the 'point' are shifted about sqrt(2)*radius
        
        x0 = point[0]-shift
        y0 = point[1]-shift
        rec_points = generate_grid_points(4, 4, pitch_recsites,x0,y0)
        MEA_dict[el] = np.array(rec_points)
        
       
        el = el+1
        
        
    return MEA_dict


def get2D_rnd_coordinates(N,c_min,c_max,sed):
    """
    Generates N random 2D coordinates (x, y) within the range [c_min,c_max).

    Args:
        N: The number of coordinates to generate.
        c_min,c_max: min and ma coordinates in micrometers
    Returns:
        A list of tuples, where each tuple represents a (x, y) coordinate.
    """
    import random
    random.seed(sed)
    coordinates = []
    for _ in range(N):
        x = random.uniform(c_min, c_max)  # Generates a float between c_min (inclusive) and c_max (exclusive)
        y = random.uniform(c_min, c_max)
        coordinates.append((x, y))
    return np.array(coordinates)


# -----------------------------------------------------------------------------
# Wallach/De Pittà 2014-style topology helpers (numpy-only, importable from
# HPC_single_run.build_topology and from the Brian2 fallback path in
# Astrocyte_Group / Synapse_to_astro). These DO NOT touch Brian2 — they only
# manipulate numpy arrays of (x, y) coordinates.
#
# Why these live here rather than in HPC_single_run.py:
#   ASD_fun_BD_cpp.py is the lower-level module imported by HPC_single_run.py
#   (no circular dependency that way), and Astrocyte_Group / Synapse_to_astro
#   need to call the same code path when used outside the HPC two-pass
#   pipeline.
# -----------------------------------------------------------------------------

def voronoi_astro_gj_pairs(N_pos, A_pos, gj_max_dist=150.0):
    """
    Build GJC adjacency between astrocytes from the JOINT Voronoi tessellation
    of neurons + astrocytes (Wallach et al., 2014; De Pittà & Berry, 2019).

    Two astrocytes are GJC-coupled iff:
      1. their Voronoi cells share a border in the joint (neurons ∪ astrocytes)
         tessellation — equivalently, they are connected by a Delaunay edge in
         the joint point set;
      2. the centre-to-centre distance does not exceed `gj_max_dist`.

    Condition (2) is the soft cap that filters out long-range "spurious"
    Voronoi neighbours arising in sparse or boundary regions, where the
    Voronoi adjacency would otherwise stretch across implausible distances
    (cf. Lallouette et al., 2014: d_max = 150 µm).

    Parameters
    ----------
    N_pos : (Nn, 2) array
        Neuron (x, y) positions in µm.
    A_pos : (Na, 2) array
        Astrocyte (x, y) positions in µm.
    gj_max_dist : float
        Soft cap, in µm. Set to numpy.inf to disable.

    Returns
    -------
    GJ_i, GJ_j : 1-D int32 arrays
        Bidirectional pair list (each undirected edge appears as both (i, j)
        and (j, i)), matching the convention used by the legacy
        `astrocyte_connections` function.
    """
    from scipy.spatial import Delaunay

    Nn = int(len(N_pos))
    Na = int(len(A_pos))

    # Degenerate cases — Delaunay needs at least 3 non-collinear points.
    if Na < 2 or (Nn + Na) < 3:
        return (np.array([], dtype=np.int32), np.array([], dtype=np.int32))

    # Joint point cloud: neurons first (global indices 0..Nn-1),
    # then astrocytes (global indices Nn..Nn+Na-1).
    points = np.vstack([N_pos, A_pos])

    try:
        tri = Delaunay(points)
    except Exception:
        # Falls back to empty GJ network if the point set is degenerate
        # (e.g. all collinear). Caller is responsible for noticing.
        return (np.array([], dtype=np.int32), np.array([], dtype=np.int32))

    # Extract undirected edges from the simplex list (each simplex is a
    # triangle in 2D, contributing 3 edges; duplicates collapse via the set).
    edge_set = set()
    for simplex in tri.simplices:
        a, b, c = int(simplex[0]), int(simplex[1]), int(simplex[2])
        edge_set.add((min(a, b), max(a, b)))
        edge_set.add((min(b, c), max(b, c)))
        edge_set.add((min(a, c), max(a, c)))

    # Keep only astrocyte–astrocyte edges, convert to local astro indices,
    # apply the soft distance cap.
    src, tgt = [], []
    for (gi, gj) in edge_set:
        if gi < Nn or gj < Nn:
            continue                          # at least one endpoint is a neuron
        ai = gi - Nn
        aj = gj - Nn
        d = float(np.hypot(A_pos[ai, 0] - A_pos[aj, 0],
                           A_pos[ai, 1] - A_pos[aj, 1]))
        if d <= gj_max_dist:
            # Emit both directions to match the existing GJ convention.
            src.append(ai); tgt.append(aj)
            src.append(aj); tgt.append(ai)

    if not src:
        return (np.array([], dtype=np.int32), np.array([], dtype=np.int32))

    return (np.asarray(src, dtype=np.int32),
            np.asarray(tgt, dtype=np.int32))


def nearest_astro_for_synapse(bouton_pos, A_pos, stoa_cutoff=70.0):
    """
    Assign each synaptic bouton to AT MOST ONE astrocyte — the nearest one
    within `stoa_cutoff` µm.

    This enforces a strict 1-synapse-to-1-astrocyte mapping at the bouton
    level (the reverse direction, many synapses → one astrocyte, is
    unconstrained, as is biologically expected: a single astrocyte
    ensheathes thousands of synapses).

    Distance is measured from the BOUTON position (S_x_syn, S_y_syn), not
    from any neuronal soma.

    Parameters
    ----------
    bouton_pos : (n_syn, 2) array
        Bouton (x, y) positions in µm.
    A_pos : (Na, 2) array
        Astrocyte (x, y) positions in µm.
    stoa_cutoff : float
        Hard distance cutoff in µm. Synapses whose nearest astrocyte lies
        farther than this are left unassigned (no StoA link).

    Returns
    -------
    StoA_i, StoA_j : 1-D int32 arrays
        Synapse → astrocyte pair list. By construction `StoA_i` contains no
        duplicates (each synapse appears at most once).
    """
    from scipy.spatial import cKDTree

    n_syn = int(len(bouton_pos))
    Na    = int(len(A_pos))
    if n_syn == 0 or Na == 0:
        return (np.array([], dtype=np.int32), np.array([], dtype=np.int32))

    tree = cKDTree(A_pos)
    # k=1 nearest neighbour for each bouton.
    dists, idx_nearest = tree.query(bouton_pos, k=1)

    keep = dists <= float(stoa_cutoff)
    syn_idx   = np.where(keep)[0].astype(np.int32)
    astro_idx = idx_nearest[keep].astype(np.int32)

    return (syn_idx, astro_idx)


def get_Raster(Traces,fs,low_f=200,high_f=2000,Visible=True):
    from scipy import signal

    from scipy.signal import find_peaks
    '''
    Alternatively an elliptic filter can be used.
    Elliptic filters offer the steepest possible rolloff between the passband and stopband for a given filter order.
    This makes them highly efficient for applications that require a sharp frequency cutoff. 
    However, this superior performance comes at the cost of ripples in both the passband and the stopband.
    
    
    Spike timings are defined in seconds
    
    Raster_array: 1st column channel 2nd column spk timings in SAMPLES
    
    '''
    
        # set up a filter to filter the voltage signal


    Wn = [2*low_f/fs, 2*high_f/fs]

    b, a = signal.butter(2, Wn, btype='bandpass')
    
    APs_time = []
    APs_unit = []
    # voltagetraces = zeros((len(Traces),len(Traces[0])))
    Raster = zeros((len(Traces),len(Traces[0])))
    Voltagefilt_array = []
    for k in range(len(Traces)):
        Trace_temp = Traces[k]
        # Subtract the mean
        Trace_temp = Trace_temp - np.mean(Trace_temp)
        Voltagefilt = signal.filtfilt(b, a, Trace_temp)  # filter
        Voltagefilt_array.append(Voltagefilt)
        threshold = np.mean(Voltagefilt) +  4 * np.std(Voltagefilt)      #threshold to detect APs
        APstemp, _ = find_peaks(abs(Voltagefilt), height=threshold)
        for j in range(len(APstemp)):
            
            
            
            APs_time = np.append(APs_time, APstemp[j])
            APs_unit = np.append(APs_unit,k)
        # voltagetraces[k, :] = Voltagefilt

       
        
        Raster[k,APstemp] = 1
        
        
    if Visible:
        t_vec = np.linspace(0,len(Traces[0]),len(Traces[0]))
        
        plt.figure()
        tertiary_color_palette = [
            # Warm Tones
            (1.0, 0.647, 0.0),    # Orange (RGB 255, 165, 0)
            (1.0, 0.498, 0.314),  # Coral (RGB 255, 127, 80)
            (0.8, 0.0, 0.0),      # Dark Red / Maroon-ish (RGB 204, 0, 0) - Not pure Red (1,0,0)
            (0.627, 0.322, 0.176),# Sienna (RGB 160, 82, 45) - Earthy Brown
            (1.0, 0.753, 0.796),  # Pink (RGB 255, 192, 203)
        
            # Cool Tones
            (0.294, 0.0, 0.510),  # Indigo (RGB 75, 0, 130) - Deep Blue-Purple
            (0.502, 0.0, 0.502),  # Purple (RGB 128, 0, 128) - More vibrant Purple
            (0.251, 0.878, 0.816),# Turquoise (RGB 64, 224, 208) - Blue-Green
            (0.0, 0.502, 0.502),  # Teal (RGB 0, 128, 128)
        
            # Earthy/Muted Tones
            (0.502, 0.502, 0.0),  # Olive (RGB 128, 128, 0) - Muted Yellow-Green
            (0.439, 0.502, 0.565),# Slate Gray (RGB 112, 128, 144) - Muted Blue-Gray
            (0.753, 0.753, 0.0)   # Chartreuse (RGB 192, 192, 0) - Muted Yellow-Green
        ]
    
        col = 0
        for ch in range(12):
            
            # if ch == 9:
            #     plt.plot(t_vec,Traces[ch]*0.1-np.mean(Traces[el])+ch*0.1,color = tertiary_color_palette[col])
            #     col = col+1
                
            # else:
            plt.plot(t_vec/fs,Voltagefilt_array[ch]-np.mean(Voltagefilt_array[ch])+ch,color = tertiary_color_palette[col])
            col = col+1

            
            indices = [i for i, x in enumerate(APs_unit) if x == ch]
        
            # Plot the unit indices (y-axis) against the spike times (x-axis)
            plt.scatter(APs_time[indices]/fs, APs_unit[indices]+0.05, s=5, marker='|',color = tertiary_color_palette[ch])
            
        # Customize the plot
        plt.title('Spiking Activity on filtered signal(Raster Plot)')
        plt.xlabel('Time (s)')
        plt.ylabel('Channel')
        
        plt.grid(True)
        plt.show()
        
    
    # Use zip to pair the elements from the two lists
    combined_list = list(zip(APs_unit, APs_time))
    
    
    # Convert the list of tuples to a numpy array
    Raster_array = np.array(combined_list)
            
                
    
    return Raster,Raster_array




def Plot_CultureDevice(Grid,Neuron_group,Nn):
    
    tertiary_color_palette = [
    # Warm Tones
    (1.0, 0.647, 0.0),    # Orange (RGB 255, 165, 0)
    (1.0, 0.498, 0.314),  # Coral (RGB 255, 127, 80)
    (0.8, 0.0, 0.0),      # Dark Red / Maroon-ish (RGB 204, 0, 0) - Not pure Red (1,0,0)
    (0.627, 0.322, 0.176),# Sienna (RGB 160, 82, 45) - Earthy Brown
    (1.0, 0.753, 0.796),  # Pink (RGB 255, 192, 203)

    # Cool Tones
    (0.294, 0.0, 0.510),  # Indigo (RGB 75, 0, 130) - Deep Blue-Purple
    (0.502, 0.0, 0.502),  # Purple (RGB 128, 0, 128) - More vibrant Purple
    (0.251, 0.878, 0.816),# Turquoise (RGB 64, 224, 208) - Blue-Green
    (0.0, 0.502, 0.502),  # Teal (RGB 0, 128, 128)

    # Earthy/Muted Tones
    (0.502, 0.502, 0.0),  # Olive (RGB 128, 128, 0) - Muted Yellow-Green
    (0.439, 0.502, 0.565),# Slate Gray (RGB 112, 128, 144) - Muted Blue-Gray
    (0.753, 0.753, 0.0)   # Chartreuse (RGB 192, 192, 0) - Muted Yellow-Green
]
    
    
    fig, ax = plt.subplots() # This creates both a figure and an axes for you
    # %matplotlib
    deep_pink = (1.0000, 0.0784, 0.5765)
    normalized_gold_rgb = (1.0, 215 / 255, 0.0)
    radius_el = 15 #[um]
    radius_cell = 9 #[um]
    
    
    col = 0
    for point in Grid:
        # point will be an array like [x, y]
        circle = plt.Circle((point[0], point[1]), radius_el, color=tertiary_color_palette[col], fill=True)
        
        ax.add_patch(circle) # Add each circle patch to the axes
        col = col+1
        
        
        
    x = Neuron_group.x
    y = Neuron_group.y
    for neu in range(Nn):
        
        circle = plt.Circle((x[neu]/um, y[neu]/um), radius_cell, color='k', fill=True)
        ax.add_patch(circle) # Add each circle patch to the axes
        
    ax.set_aspect('equal', adjustable='box')
    min_x = np.min(Grid[:, 0]) - radius_el * 1.2 # Add some buffer
    max_x = np.max(Grid[:, 0]) + radius_el * 1.2
    min_y = np.min(Grid[:, 1]) - radius_el * 1.2
    max_y = np.max(Grid[:, 1]) + radius_el * 1.2

    ax.set_xlim(min_x, max_x)
    ax.set_ylim(min_y, max_y)

    plt.xlabel("[um]")
    plt.ylabel("[um]")
    plt.show()  
    
    
def Electrode_traces(pitch,pitch_recsites,shift,N,MonitorN,electrode_dist,neuron_radius,electrode_radius,Visible = False):
    
    '''
    MEA_dict = dictionary with electrodes info: position and recordin sites
    Traces = List of Lists that contain the recordings.
    
    
    '''
    
    
    
    Grid = Get_12grid(pitch)
    
    MEA_dict = Recording_sites(pitch_recsites,shift,Grid)
    
    # --- Plot Device + Neurons
    
    if Visible:
        
        Plot_CultureDevice(Grid,N,N.N)
    
    Traces = Electrode_recording(MEA_dict,N,MonitorN,electrode_dist,neuron_radius,electrode_radius)
    
    if Visible:
        t_vec = np.linspace(0,len(Traces[0]),len(Traces[0]))
        
        plt.figure()
        tertiary_color_palette = [
            # Warm Tones
            (1.0, 0.647, 0.0),    # Orange (RGB 255, 165, 0)
            (1.0, 0.498, 0.314),  # Coral (RGB 255, 127, 80)
            (0.8, 0.0, 0.0),      # Dark Red / Maroon-ish (RGB 204, 0, 0) - Not pure Red (1,0,0)
            (0.627, 0.322, 0.176),# Sienna (RGB 160, 82, 45) - Earthy Brown
            (1.0, 0.753, 0.796),  # Pink (RGB 255, 192, 203)
        
            # Cool Tones
            (0.294, 0.0, 0.510),  # Indigo (RGB 75, 0, 130) - Deep Blue-Purple
            (0.502, 0.0, 0.502),  # Purple (RGB 128, 0, 128) - More vibrant Purple
            (0.251, 0.878, 0.816),# Turquoise (RGB 64, 224, 208) - Blue-Green
            (0.0, 0.502, 0.502),  # Teal (RGB 0, 128, 128)
        
            # Earthy/Muted Tones
            (0.502, 0.502, 0.0),  # Olive (RGB 128, 128, 0) - Muted Yellow-Green
            (0.439, 0.502, 0.565),# Slate Gray (RGB 112, 128, 144) - Muted Blue-Gray
            (0.753, 0.753, 0.0)   # Chartreuse (RGB 192, 192, 0) - Muted Yellow-Green
        ]
    
        col = 0
        for ch in range(12):
            
            # if ch == 9:
            #     plt.plot(t_vec,Traces[ch]*0.1-np.mean(Traces[el])+ch*0.1,color = tertiary_color_palette[col])
            #     col = col+1
                
            # else:
                plt.plot(t_vec,Traces[ch]-np.mean(Traces[ch])+ch*0.1,color = tertiary_color_palette[col])
                col = col+1
        plt.show()
        
    
    return Traces,MEA_dict







#---------------------------------------- NEURONAL DYNAMICS ----------------------------------------
# --------------------------------- DATA PREPROCESSING ---------------------------------   




def Standardization(data):
    """
    Standardizes a time series (univariate or multivariate) by standardizing each feature (column) separately.

    Standardization (Z-score normalization) transforms the data to have a mean of 0 and a standard deviation of 1.
    The formula for standardization is: z = (x - mu) / sigma, where mu is the mean and sigma is the standard deviation.

    Args:
        data (np.ndarray): A NumPy array representing the time series.
                          It can be 1D (univariate) or 2D (multivariate) of shape (timesteps, features).

    Returns:
        np.ndarray: A NumPy array of the same shape as `data`, but with each feature standardized.
                    Returns None if the input data is not a 1D or 2D array.
    """
    

    

   
    mean = np.mean(data)
    std_dev = np.std(data)
    
    
    return (data-mean)/std_dev

        # Handle the case where the standard deviation is zero
      
def Get_IFR(data, fs, Cumulative, t_vec, step_s, bin_size, Isolate_NB, T_max):
    """
    Calculates the Instantaneous Firing Rate (IFR) of the neuronal data.

    Args:
        data (list): A list of spike timings for each channel.
        fs (int): Sampling frequency in Hz.
        Cumulative (numpy.ndarray): The cumulative global activity.
        t_vec (numpy.ndarray): The time vector for the cumulative activity.
        step_s (float): Step size for the cumulative activity calculation.
        bin_size (float): Bin size in seconds.
        Isolate_NB (bool): If True, isolates IFR for neurobursts.
        T_max (int): Total recording time in samples.

    Returns:
        tuple: A tuple containing:
            IFR (list or numpy.ndarray): The calculated IFR.
            bin_size (float): The bin size in samples.
            window_size (int): The window size for NB analysis in samples,
                               or None if Isolate_NB is False.
    """
    bin_size_samples = int(bin_size * fs)  # [samples]

    if Isolate_NB:
        # Construct the window that will be centered at the NB's peak
        pre_w = 1.5  # Pre samples [s]
        post_w = 3.5 # post samples [s]

        # Scale to samples
        pre_w_samples = int(pre_w * fs)
        post_w_samples = int(post_w * fs)

        # Define the window size for calculating the IFR
        window_size = pre_w_samples + post_w_samples  # [samples]
        num_bins = window_size // bin_size_samples

        # Isolate the NB timings (peaks location).
        mean_IFR = np.mean(Cumulative)
        std_IFR = np.std(Cumulative)
        
        # The 'distance' argument for find_peaks is in samples, not seconds.
        # It should be based on the sampling of 'Cumulative', which is 'step_s'.
        # The MATLAB code uses 3 * fs / step_s, where fs is the original sampling rate.
        # This seems to be a scaling factor. We'll replicate it.
        min_peak_distance_samples = int(3 * fs / step_s)
        
        # 'height' in scipy.signal.find_peaks is the equivalent of MinPeakHeight
        idx, _ = find_peaks(Cumulative.flatten(), height=mean_IFR + std_IFR, distance=min_peak_distance_samples)
        NB_T = t_vec[idx]

        IFR = [None] * len(NB_T)
        num_channels = len(data)

        for nb_idx, nb_time in enumerate(NB_T):
            binned_NB = np.zeros((num_channels, num_bins))

            lower_bound = nb_time - pre_w_samples
            upper_bound = nb_time + post_w_samples

            # Extract per each channel the spikes within the NB window
            for ch_idx in range(num_channels):
                data_ = data[ch_idx]

                # Bins for this specific NB
                for bin_idx in range(num_bins - 1):
                    # Define the start and stop timings
                    start_idx = int(lower_bound + bin_idx * bin_size_samples)
                    stop_idx = int(lower_bound + (bin_idx + 1) * bin_size_samples)

                    # Extract spikes within the window (inclusive of the boundaries)
                    within_range = (data_ >= start_idx) & (data_ < stop_idx)

                    # Count and store the number of spikes
                    number_spks = np.sum(within_range)
                    binned_NB[ch_idx, bin_idx] = number_spks

            IFR[nb_idx] = binned_NB

    else:
        window_size = None
        num_channels = int(len(data))
        num_bins = int(T_max // bin_size_samples)
        IFR = np.zeros((num_bins, num_channels))

        for ch_idx in range(num_channels):
            data_ = data[ch_idx]

            for bin_idx in range(num_bins):
                start_idx = bin_idx * bin_size_samples
                stop_idx = (bin_idx + 1) * bin_size_samples
                
                within_range = (data_ >= start_idx) & (data_ < stop_idx)
                
                number_spks = np.sum(within_range)
                
                IFR[bin_idx, ch_idx] = number_spks

    return IFR, bin_size_samples, window_size


def Rect_window(fs, w_size_s, overlap_s, x, T_max):
    """
    Calculates cumulative activity using a sliding rectangular window.

    Args:
        fs (int): Sampling frequency in Hz.
        w_size_s (float): Window size in seconds.
        overlap_s (float): Overlap between windows in seconds.
        x (list): A list of spike timings for each channel.
        T_max (int): Total recording time in samples.

    Returns:
        tuple: A tuple containing:
            Cumulative (numpy.ndarray): The cumulative activity. NOT NORMALIZED
            t_vec (numpy.ndarray): The time vector for the cumulative activity.
            step_size (int): The step size between windows in samples.
    """
    w_size = int(w_size_s * fs)
    overlap = int(overlap_s * fs)
    
    step_size = w_size - overlap
    
    # In MATLAB, '0:step_size:T_max' is inclusive, so we need to adjust np.arange.
    t_vec = np.arange(0, T_max, step_size)
    Cumulative = np.zeros(len(t_vec))
    
    start_index = 0
    idx = 0
    while start_index + w_size <= T_max and idx < len(t_vec):
        temp_cum = 0
        for ch in x:
            data_ = ch
            
            # The MATLAB code uses start_index + w_size-1, which is correct for 1-based indexing.
            # For Python, we use the end index exclusively.
            within_range = (data_ >= start_index) & (data_ < start_index + w_size)
            
            count = np.sum(within_range)
            
            temp_cum += count
            
        # Cumulative[idx] = temp_cum / w_size
        Cumulative[idx] = temp_cum
        
        start_index += step_size
        idx += 1
        
    return Cumulative, t_vec, step_size
    
def get_PCA(NB_IFR_smoothed_concatenated, IFR_smoothed, Isolate_NB,Visible):
    """
    Performs Principal Component Analysis (PCA) on the IFR data.

    Args:
        NB_IFR_smoothed_concatenated: The concatenated smoothed IFR data.
                                      If Isolate_NB is True, this is used for PCA.
        IFR_smoothed: The smoothed IFR data, which can be a list of arrays (Isolate_NB=True)
                      or a 2D array (Isolate_NB=False).
        Isolate_NB: A boolean indicating whether to perform PCA on concatenated
                    neuroburst (NB) data or the total signal.
    
    Returns:
        Variance_explained: The percentage of variance explained by each PC.
        Projected_trajectories: The data projected onto the new PCA space.
        Coefficients: The principal component coefficients (eigenvectors).
        NB_IFR_PCA_mean: The mean PCA trajectory (if Isolate_NB is True).
    """

    if Isolate_NB:
        # In scikit-learn's PCA, the input data should have shape (n_samples, n_features).
        # MATLAB's pca assumes rows are observations and columns are variables.
        # So we transpose the concatenated data.
        NB_IFR_smoothed_concatenated_PCA = NB_IFR_smoothed_concatenated.T

        # Perform PCA
        pca = PCA()
        pca.fit(NB_IFR_smoothed_concatenated_PCA)
        Coefficients = pca.components_.T
        Variance_explained = pca.explained_variance_ratio_

        # Obtain the mean traces
        num_nb = len(IFR_smoothed)
        num_channels = IFR_smoothed[0].shape[0] if num_nb > 0 else 0
        samples_per_window = IFR_smoothed[0].shape[1] if num_nb > 0 else 0
        
        NB_IFR_PCA_mean_ = np.zeros((num_channels, samples_per_window, num_nb))

        for k in range(num_nb):
            nb_ifr_pca_data = IFR_smoothed[k]
            # Project the data
            proj = nb_ifr_pca_data.T @ Coefficients
            NB_IFR_PCA_mean_[:, :, k] = proj.T
        
        # Calculate the mean across the third dimension (k)
        NB_IFR_PCA_mean = np.mean(NB_IFR_PCA_mean_, axis=2).T
        
        # Project the concatenated data
        Projected_trajectories = NB_IFR_smoothed_concatenated_PCA @ Coefficients

        # Plotting
        fig = plt.figure()
        ax = fig.add_subplot(111, projection='3d')
        
        # Extract the first three components
        x = Projected_trajectories[:, 0]
        y = Projected_trajectories[:, 1]
        z = Projected_trajectories[:, 2]

        x_m = NB_IFR_PCA_mean[:, 0]
        y_m = NB_IFR_PCA_mean[:, 1]
        z_m = NB_IFR_PCA_mean[:, 2]
        
        # Use plot3 function to plot the lines
        ax.plot(x, y, z)
        ax.plot(x_m, y_m, z_m, linewidth=4.5, color='red')
        
        ax.set_xlabel('PC 1')
        ax.set_ylabel('PC 2')
        ax.set_zlabel('PC 3')
        ax.set_title('Concatenated NBs')
        
        # To keep the axes scaled appropriately and prevent distortion
        ax.set_box_aspect([1, 1, 1])  # equal aspect ratio
        
        ax.grid(True)
        plt.show()

    else:
        
        
        # In this case, Isolate_NB is false and IFR_smoothed is a 2D array.
        NB_IFR_PCA_mean = None
        
        # Perform PCA
        pca = PCA()
        pca.fit(IFR_smoothed)
        Coefficients = pca.components_.T
        Variance_explained = pca.explained_variance_ratio_
        
        # Project the data
        Projected_trajectories = pca.transform(IFR_smoothed)


        if Visible == True:
            # Plotting
            fig = plt.figure()
            ax = fig.add_subplot(111, projection='3d')
            
            # Extract the first three components
            x = Projected_trajectories[:, 0]
            y = Projected_trajectories[:, 1]
            z = Projected_trajectories[:, 2]
            
            # Use plot3 function to plot the lines
            ax.plot(x, y, z)
            
            ax.set_xlabel('PC 1')
            ax.set_ylabel('PC 2')
            ax.set_zlabel('PC 3')
            ax.set_title('Culture Dynamics')
            
            ax.set_box_aspect([1, 1, 1])
            
            ax.grid(True)
            plt.show()

    return Variance_explained, Projected_trajectories, Coefficients, NB_IFR_PCA_mean




def Smoothed_IFR(IFR, bin_size, window_size, fs, Isolate_NB, Gaussian_window, Visible):
    """
    The function takes the raw instantaneous firing rates of the NB-centered
    windows and returns the concatenated and smoothed NB's IFR.
    
    Args:
        IFR: The input IFR data. Its format depends on Isolate_NB.
        bin_size: The size of the time bins.
        window_size: The size of the analysis window.
        fs: The sampling frequency.
        Isolate_NB: If True, IFR is a list of arrays (cell array in MATLAB).
                    If False, IFR is a 2D NumPy array.
        Gaussian_window: The size of the Gaussian smoothing window [s].
        Visible: A boolean to control whether to display plots.
    
    Returns:
        IFR_smoothed: The smoothed IFR data.
        IFR_smoothed_concatenated: The concatenated smoothed IFR data.
    """
    Gaussian_window_samples = Gaussian_window*fs
    if Isolate_NB:
        # MATLAB uses 1-based indexing for size, Python uses 0-based
        num_nb = len(IFR)
        num_channels = IFR[0].shape[0] if num_nb > 0 else 0
        
        # Calculate samples_per_window
        # Assuming Samples_per_window is a global variable in the MATLAB code,
        # we'll calculate it here from the input IFR data.
        if num_nb > 0:
            samples_per_window = IFR[0].shape[1]
        else:
            samples_per_window = 0

        # Create time vector for plotting
        t_vec_nb = np.arange(0, samples_per_window) * bin_size

        if Visible:
            plt.figure()
            for j in range(num_nb):
                ifr_data = IFR[j]
                for i in range(num_channels):
                    channel = ifr_data[i, :]
                    plt.plot(t_vec_nb, channel)
            plt.title('Raw IFR')
            plt.xlabel('Time [s]')
            plt.ylabel('Spikes')
            plt.show()

        IFR_smoothed = [None] * num_nb
        IFR_smoothed_concatenated = np.zeros((num_channels, num_nb * samples_per_window))
        
        # MATLAB's smoothdata('gaussian') is equivalent to a Gaussian filter.
        # We'll use scipy.ndimage.gaussian_filter1d for this.

        for j in range(num_nb):
            ifr_data = IFR[j]
            smoothed_channels = []
            for i in range(num_channels):
                channel = ifr_data[i, :]
                smoothed_channel = gaussian_filter1d(channel.astype(float), sigma=Gaussian_window_samples)
                smoothed_channels.append(smoothed_channel)
                
                # Concatenate the smoothed data
                # MATLAB's n*Samples_per_window + 1 : (n+1)* Samples_per_window
                # is equivalent to n*samples_per_window : (n+1)* samples_per_window in Python
                IFR_smoothed_concatenated[i, j * samples_per_window : (j + 1) * samples_per_window] = smoothed_channel

            IFR_smoothed[j] = np.array(smoothed_channels)

        if Visible:
            plt.figure()
            plt.subplot(2, 1, 1)
            for j in range(num_nb):
                ifr_data = IFR_smoothed[j]
                for i in range(num_channels):
                    channel = ifr_data[i, :]
                    plt.plot(t_vec_nb, channel)
            plt.title(f'Smoothed IFR')
            plt.xlabel('Time [s]')
            plt.ylabel('Spikes')

            plt.subplot(2, 1, 2)
            # The original code plots all channels; Ch = 3 is not used.
            # We'll follow the original code and plot all.
            t_vec_conc = np.arange(IFR_smoothed_concatenated.shape[1])
            plt.plot(t_vec_conc, IFR_smoothed_concatenated.T)
            plt.title(f'Concatenated smoothed NB IFR')
            plt.xlabel('Samples')
            plt.ylabel('Spikes')
            plt.tight_layout()
            plt.show()

    else:
        # IFR is a 2D array: samples x channels
        num_samples, num_channels = IFR.shape
        IFR_smoothed = np.zeros_like(IFR)
        IFR_smoothed_concatenated = []

        if Visible:
            plt.figure()
            plt.subplot(2, 1, 1)
            for i in range(num_channels):
                plt.plot(IFR[:, i])
            plt.title(f'Raw IFR')
            plt.xlabel('Samples')
            plt.ylabel('Spikes')
            
            plt.subplot(2, 1, 2)
            for i in range(num_channels):
                channel = IFR[:, i]
                smoothed_channel = gaussian_filter1d(channel.astype(float), sigma=Gaussian_window_samples)
                IFR_smoothed[:, i] = smoothed_channel
                plt.plot(smoothed_channel)
            plt.title(f'Smoothed IFR')
            plt.xlabel('Samples')
            plt.ylabel('Spikes')
            plt.tight_layout()
            plt.show()
        else:
            for i in range(num_channels):
                channel = IFR[:, i]
                IFR_smoothed[:, i] = gaussian_filter1d(channel.astype(float), sigma=Gaussian_window_samples)
            
    return IFR_smoothed, IFR_smoothed_concatenated

        

def get_Smoothed_Cumulative(Cumulative,fs_downsampled,Gaussian_window):
    # Gaussian window is the std of the gaussian window. it is defined in s
    # and MUST be grater than the sampling step of fs_downsampled

    
    # Gaussian window is defined in s, thus devide by 1000 because fs_downsampled is in Hz
    Gaussian_window_samples = np.ceil(Gaussian_window*fs_downsampled) 
    
    
    if  Gaussian_window_samples == 1:
        
        raise ValueError("Single sample window width.")
    
    # Check consistency
    if Gaussian_window_samples <=5: # five samples are not a lot
    
    
        print('Smoothing with a narrow gaussian window...')
        
        
    smoothed_cumulative = gaussian_filter1d(Cumulative.astype(float), sigma=Gaussian_window_samples)

        
        
    
    
    return smoothed_cumulative
    
        

def calculate_mean_burst_duration(time_series_data, fs,scal_factor = 0.5, Visible=False):
    """
    Calculates the mean duration of bursts in a time series of network data and can plot the results.

    Args:
        time_series_data (list or np.array): The network data time series.
        fs = [Hz]
        baseline (float): The threshold value that defines a burst.
        plot (bool): If True, a plot of the data with burst start/end points is created.

    Returns:
        float: The mean duration of all detected bursts. Returns 0 if no bursts are found. the unit of time is s
    """
    burst_durations = []
    in_burst = False
    current_burst_duration = 0
    baseline = np.mean(time_series_data)*scal_factor
    time_step = 1/fs
    burst_start_indices = []
    burst_end_indices = []

    for i, data_point in enumerate(time_series_data):
        if data_point > baseline:
            if not in_burst:
                in_burst = True
                current_burst_duration = time_step
                burst_start_indices.append(i)
            else:
                current_burst_duration += time_step
        else:
            if in_burst:
                burst_durations.append(current_burst_duration)
                in_burst = False
                current_burst_duration = 0
                burst_end_indices.append(i - 1)

    if in_burst:
        burst_durations.append(current_burst_duration)
        burst_end_indices.append(len(time_series_data) - 1)

    if Visible:
        time_points = [i /fs for i in range(len(time_series_data))]
        plt.figure(figsize=(12, 6))
        plt.plot(time_points, time_series_data, label='Global activity')
        plt.axhline(y=baseline, color='r', linestyle='--', label=f'Baseline ({baseline})')

        start_time_points = [time_points[i] for i in burst_start_indices]
        start_values = [time_series_data[i] for i in burst_start_indices]
        end_time_points = [time_points[i] for i in burst_end_indices]
        end_values = [time_series_data[i] for i in burst_end_indices]

        plt.scatter(start_time_points, start_values, color='g', marker='o', s=100, label='Burst Start')
        plt.scatter(end_time_points, end_values, color='b', marker='x', s=100, label='Burst End')

        plt.title(f'Global activity. MBD: {np.mean(burst_durations)}')
        plt.xlabel(f'Time [s])')
        plt.ylabel(f'Global activity')
        plt.legend()
        plt.grid(True)

    
    if not burst_durations:
        return 0
    return np.mean(burst_durations)  



def Neuronal_traces_simulation(Raster_array,Type ='Cumulative',t_rec = 600, fs = 10000, w_size=0.02, overlap = 0.06, 
                    bin_size_s = 0.05, Isolate_NB = False,Gaussian_window=0.04,
                     Visible = True,NB_statistics = False,Normalization_type = 'Peak amplitude'):
    
    # Raster_array = nx2, 1st column the channel's idx, 2nd column the timing of spike in seconds
    # Type = PCA or Cumulative. PCA = Usual neuronal dynamics, Cumulative= Cumulative IFR on all the electrodes.
    # t_rec = 600  # [s] Recording time
    # fs = 10000


    # Normalization = 'Standardization' or 'Peak amplitude' type of normalization

    # Visible = True

    # # Calculate the GA
    # w_size = 0.12  # [s] 12
    # overlap = 0.06  # [s]

    # # Bin size for the IFR
    # bin_size_s = 0.05  # [s] 0.005 = 5 [ms]

    # # Whether to Isolate NB or keep the total signal
    # Isolate_NB = False

    # # Window size for smoothing
    # Gaussian_window = 2  # [s]

    # # Extract data
    # data = [None] * len(Strings)
    
    # Extract data    
    n_channels = int(np.max(Raster_array[:,0]) + 1)
   
    data = [None] * n_channels
    
    T_max = t_rec * fs
    
    # COnvert the sigma of the gaussian window in samples
    
    
    
    
    
    for i in range(n_channels):
        # find non-zero elements
        spk_timing = np.where(Raster_array[:,0]  == i)[0]
        data[i] = Raster_array[spk_timing,1]

    
    if Visible:
        plt.figure()
        for i in range(n_channels):
            data_timings = data[i]
            data_plot = np.ones(len(data_timings)) * (i + 1)
            plt.scatter(data_timings, data_plot, s=15, marker='.')
        plt.xlabel('Time [s]')
        plt.ylabel('Electrodes')
        plt.title('Spike Timings')
        plt.grid(True)
        plt.show()
    
        # Calculate NBs
        # You would need to define Rect_window in Python
        # [Cumulative, t_vec, step_s] = Rect_window(fs, w_size, overlap, data, T_max)
        # The following is a placeholder for the Rect_window function call
        # This part would need to be implemented in Python based on the MATLAB function's logic
        
        # Assume Cumulative, t_vec, and step_s are computed here
        # For example:
        # Cumulative, t_vec, step_s = Rect_window(fs, w_size, overlap, data, T_max)
    
        # # Let's assume we have Cumulative, t_vec, and step_s for the next part
        # # Example mock data for plotting:
        # t_vec = np.linspace(0, t_rec, int(T_max))
        # Cumulative = np.random.rand(len(t_vec)) * 100
    
        # Mean_IFR = np.mean(Cumulative)
        # STD_IFR = np.std(Cumulative)
        # plot_MFR = np.ones(len(t_vec)) * Mean_IFR
        # plot_MFR_STD_plus = np.ones(len(t_vec)) * (Mean_IFR + STD_IFR)
        # plot_MFR_STD_minus = np.ones(len(t_vec)) * (Mean_IFR - STD_IFR) # The MATLAB code had a mistake here
        
        # # findpeaks
        # # The 'MinPeakDistance' argument in MATLAB is different in Python's find_peaks
        # # In Python, distance is in samples, not seconds.
        # # The MATLAB code has 3 * fs / step_s, which should be adjusted for Python
        # # Assuming step_s is the sampling rate of Cumulative, not fs
        
        # # Let's assume a step_s value
        # step_s = 1000 # Example step_s value
        # idx, _ = find_peaks(Cumulative, height=Mean_IFR + STD_IFR, distance=int(3 * fs / step_s))
        # NB_T = t_vec[idx]
        
        # plt.figure()
        # plt.plot(Cumulative)
        # plt.plot(idx, Cumulative[idx], 'x')
        # plt.title('findpeaks')
        # plt.show()
    
        # plt.figure()
        # plt.title('Global Activity')
        # plt.plot(t_vec / fs, Cumulative, label='Cumulative IFR')
        # plt.plot(t_vec / fs, plot_MFR, linestyle='-.', color='r', linewidth=1.5, label='Mean IFR')
        # plt.plot(t_vec / fs, plot_MFR_STD_plus, linestyle='--', color='g', linewidth=1.5, label='Mean + STD')
        # plt.xlabel('Time [s]')
        # plt.ylabel('Instantaneous firing rate [spk/s]')
        # plt.legend()
        # plt.show()
    
    # Calculate NBs
    # You would need to define Rect_window in Python
    if Type == 'PCA':
        [Cumulative, t_vec, step_s] = Rect_window(fs, w_size, overlap, data, T_max)
        
        [IFR, bin_size,window_size] = Get_IFR(data,fs,Cumulative,t_vec,step_s,bin_size_s,Isolate_NB,T_max);
            
   
        fs_downsampled = 1/bin_size_s
        
    
   
        [IFR_smoothed, IFR_smoothed_concatenated] = Smoothed_IFR(IFR, bin_size,window_size,fs_downsampled,Isolate_NB,Gaussian_window,Visible);
    
    
    
        [Variance_explained, Projected_trajectories,Coefficients,NB_IFR_PCA_mean] = get_PCA(IFR_smoothed_concatenated,IFR_smoothed,Isolate_NB,Visible);
    
        
    
        return Projected_trajectories,Variance_explained,fs_downsampled
    
    
    elif Type == 'Cumulative':
       overlap = 0
       [Cumulative, t_vec, step_s] = Rect_window(fs, w_size, overlap, data, T_max)
       
       # The new sampling frequency is downsampled by a factor determined by w_size [s]
       fs_downsampled = 1/w_size
       
       
       smoothed_cumulative =  get_Smoothed_Cumulative(Cumulative,fs_downsampled,Gaussian_window) 
       
       
       if Normalization_type == 'Standardization':
           print('Cumulative traces are STANDARDIZED')
           smoothed_cumulative = Standardization(smoothed_cumulative)
       
           if Visible == True:
               
               plt.figure()
               plt.plot(t_vec/fs,smoothed_cumulative,color = 'r')
               # plt.plot(t_vec,Cumulative,color = 'b')
               plt.xlabel('Time [s]')
               plt.ylabel('Standardized and Smoothed IFR')
               plt.show()
               
               
       elif Normalization_type == 'Peak amplitude':
           print('Cumulative traces are NORMALIZED')
           
           peak_amplitude = np.max(smoothed_cumulative)
           
           smoothed_cumulative = smoothed_cumulative/peak_amplitude
           if Visible == True:
               
               plt.figure()
               plt.plot(t_vec/fs,smoothed_cumulative,color = 'r')
               # plt.plot(t_vec,Cumulative,color = 'b')
               plt.xlabel('Time [s]')
               plt.ylabel('Normalized and Smoothed IFR')
               plt.show()

           
           
           
       
             
            
            
            
        
        
    return smoothed_cumulative,fs_downsampled,t_vec
        

# ------------------------------------ SYNAPSE POSITION FUNCTIONS -----------------------------------------------------

def map_range(value, from_min, from_max, to_min, to_max):
    """
    Maps a value from one range to another range using linear interpolation.

    Args:
        value: The value to be mapped.
        from_min: The minimum value of the original range.
        from_max: The maximum value of the original range.
        to_min: The minimum value of the target range.
        to_max: The maximum value of the target range.

    Returns:
        The mapped value in the target range.
    """

    # Check for valid input ranges (avoid division by zero)
    if from_max - from_min == 0:
        raise ValueError("Input range cannot be zero.")
    if to_max - to_min == 0:
      raise ValueError("Target range cannot be zero.")

    # Linear transformation formula
    mapped_value = (value - from_min) * (to_max - to_min) / (from_max - from_min) + to_min
    return mapped_value


def get_dendrite_prob(r_1,r_0,dx,map_magnitude):
    '''
    This function is thought to establish a evidence-based distance-dependent 
    connection probability field about the interested somata within the shell enacapsuled
    in 'r_0' and 'r_1', this whithout a detailed representation of the dendritic 
    arbourization. The analysis hinges on the morphometric analysis on hipsc control lines
    in the following papers:
        1) https://doi.org/10.1038/s41467-019-12947-3
        2) https://doi.org/10.1016/j.celrep.2020.107538
    
    The sholl analysis evidenced a peak in branching phenomena at approx. 40/50 um from the
    soma. Therefore the switch of neurites from primary branches to secondary ones is likely
    to happen here. This is important to set the correct neurite diameter 'Branch_d'. The mean
    dendritic length distribution across the distance from the soma is reproduced by scaling
    a Chi-squared distribution (along both axes) with 4 degrees of freedom.
    The purpose is to determine the ratio between the total volume of dendritic abourization 
    within the shell and the total volume of the latter. The total branch length is calculated 
    by computing the cumulative density function between r_0 and r_1 (integral through the trapz 
    function of the PDF within the defined range). In this way the probability of an
    axon to cross a dendritic process is defined. Given the distribution the influence region spans
    approx 200 um in radius even though the probabiities at the boundaries borders on zero.
    
    Parameters
    ----------
    r_1 : float [um]
        Outer shell radius
    r_0 : float [um]
        Inner shell radius
    dx : float 
        Spacing between sample points for the trapezoidal method
    map_magnitude : integer [um]
        Upper limit of the scaling map of the distribution's x axes.
    Returns
    -------
    Filled_volume : float 
        Probability of neurite presence in the shell [0,1]

    '''
    
    if r_1 <= 40:
        Branch_d = 2.1 # [um] # primary neurites
        # r_1 = 40
        # r_0 = 0
        
    else:
        Branch_d = 1.51 # [um] # secondary neurites
        r_1 = r_1
        r_0 = r_0
    
    
    value_ = np.arange(0,250,0.1)  #[um] Perisomatic location
    sqr = np.zeros(len(value_))
    j=0
    for i in value_:
        
        maped_val = map_range(i, 0, map_magnitude, 0, 12)    
        sqr[j] = chi2.pdf(maped_val, 4)
        j=j+1
        

    
    #%

    sqr_max = max(sqr)
    value_ = np.arange(0,250,0.1)  #[um] Perisomatic location
    sqr = np.zeros(len(value_))
    map_sqr = np.zeros(len(value_))
    j=0
    mode = 110
    for i in value_:
        
        maped_val = map_range(i, 0, map_magnitude, 0, 12)    
        sqr = chi2.pdf(maped_val, 4)
        # map_sqr[j] = sqr*714.2
        maped_val_ = map_range(sqr,0, sqr_max,0,mode)    
        map_sqr[j] = maped_val_
        j=j+1
         
    # plt.figure()
    # plt.title('Scaled Chi-squared distribution. DoF: 4')
    # plt.plot(value_,map_sqr)
    # plt.xlabel('Distance form soma [um]')
    # plt.ylabel('Mean dendritic length [um]')    

    x_v_1= np.where(value_ == r_1)[0][0]
    x_v_0= np.where(value_ == r_0)[0][0]

    x_1 = value_[x_v_0:x_v_1]
    y_1= map_sqr[x_v_0:x_v_1]
    Cumulative_length_1 = np.trapz(y_1,x_1,dx=dx)

    ##### Calculate
     
    V_sphere = (4/3)*np.pi*(r_1**3) -(4/3)*np.pi*(r_0**3)

    
    V_dendrite = (Cumulative_length_1/4)*np.pi*(Branch_d**2)

    Filled_volume = (V_dendrite/V_sphere)
    
    return Filled_volume
    
def generate_dendritic_arbour(max_rad = 220,dx=0.01,interval=1):
    '''
    

   Parameters
   ----------
   max_rad : integer. 220[um]
       Maximum radius from the soma to calculate the probability of presence of 
       a dendritic process
   dx : integer. 0.0001
       Spacing between sample points for the trapezoidal method

   interval : integer. 5
       Step in computing the shell ranges (r_0 and r_1) from 0 to max_rad.
       Must be an integer of thte latter

   Returns
   -------
   Syn_prob : array (N,)
       Array of probability of connections for a axonal neurite 
   
    rarius_val : array (N,)
        Distances from the soma where the relative elements in 'Syn_prob'
        have been calculated
   
    Where N is max_rad/interval
    '''
    

    
    
    max_rad = 220
    rarius_val = np.arange(0,max_rad,interval)
    Syn_prob = np.zeros(len(rarius_val))
    k= 0
    dx = 0.0001
    for i in np.arange(1,len(rarius_val)):
        
      r_1 = rarius_val[i]
      r_0 = rarius_val[i-1]
      Syn_prob[k] = get_dendrite_prob(r_1,r_0,dx,max_rad)
      k = k+1
      
    # plt.figure()
    # plt.title('Synapse probability')
    # plt.plot(rarius_val,Syn_prob)
    # plt.xlabel('Distance from soma [um]')
    # plt.ylabel('Probability')
    # # plt.yscale('log')
    # plt.xlim(0,200)
    
    
    
    return Syn_prob,rarius_val
        
def sample_distance_from_soma(syn_prob, rarius_val, rng):
    """
    Sample a single distance from the soma based on the provided probability
    distribution.

    Parameters
    ----------
    syn_prob : np.array
        Array of connection probabilities for each shell.
    rarius_val : np.array
        Distances from the soma corresponding to the shells.
    rng : numpy.random.Generator
        Seeded RNG passed in by the caller. Required — using an unseeded
        default generator was breaking reproducibility of synapse positions
        (Report 3 §A.4).

    Returns
    -------
    float
        A single distance value (in um) sampled from the distribution.
    """
    prob_sum = np.sum(syn_prob)
    if prob_sum == 0:
        print("Warning: all syn_prob are zero. Falling back to uniform sampling.")
        return rng.choice(rarius_val)

    normalized_probs = syn_prob / prob_sum
    valid_indices = np.where(normalized_probs > 0)[0]
    sampled_distance = rng.choice(
        rarius_val[valid_indices],
        p=normalized_probs[valid_indices],
        replace=True,
    )
    return sampled_distance


def get_synapse_coordinates(Synapse, Neuron, Syn_prob, rarius_val,
                            displ_bias=15, seed_syn=None):
    """
    Place each synaptic bouton on the dendrite of the post-synaptic neuron, a
    sampled distance from the post-soma, along the line connecting pre and
    post.

    No autaptic connections (assumed already excluded by `i != j` at
    `S.connect`).

    A constant bias `displ_bias` (in um, default 15) is added to every sample
    so the bouton never coincides with the soma centre.

    All distances are expressed in um.

    Parameters
    ----------
    Synapse : brian2.Synapses
    Neuron : brian2.NeuronGroup
    Syn_prob, rarius_val : np.array
        Dendritic-arbour PDF and its support (see generate_dendritic_arbour).
    displ_bias : float
        Constant um offset added to each sampled distance.
    seed_syn : int or None
        Seed for the RNG that samples bouton distances. Critical for
        reproducibility (Report 3 §A.4).

    Returns
    -------
    list of (x, y) tuples in um.
    """
    rng = np.random.default_rng(seed_syn)

    synapse_coords = []
    for i in range(len(Synapse)):
        pre_idx = Synapse.i[i]
        post_idx = Synapse.j[i]

        pre_pos = np.array([Neuron.x[pre_idx] / um, Neuron.y[pre_idx] / um])
        post_pos = np.array([Neuron.x[post_idx] / um, Neuron.y[post_idx] / um])

        vector = post_pos - pre_pos
        vector_length = np.linalg.norm(vector)
        unit_vector = vector / vector_length

        # Sample a distance for this synapse, retrying if it would overshoot
        # past the pre-synaptic neuron.
        distance = sample_distance_from_soma(Syn_prob, rarius_val, rng) + displ_bias
        while distance > vector_length:
            distance = sample_distance_from_soma(Syn_prob, rarius_val, rng) + displ_bias

        # Move 'distance' along the unit vector from the post-syn neuron.
        new_coord = post_pos - unit_vector * distance
        synapse_coords.append(tuple(new_coord))

    return synapse_coords
