

from mpl_toolkits.mplot3d import Axes3D
from mpl_toolkits.mplot3d.art3d import Poly3DCollection, Line3DCollection
import matplotlib.colors as mcolors
from sklearn.metrics import mean_squared_error
import numpy as np
# import torch
import scipy.io
import pdb
import os
from scipy.io import loadmat
from scipy.signal import find_peaks
from scipy.ndimage import gaussian_filter1d
from sklearn.decomposition import PCA
from mpl_toolkits.mplot3d import Axes3D

from brian2 import *
from brian2 import devices

from sklearn.neighbors import KDTree
from scipy.stats import skewnorm
from scipy.stats import chi2
import seaborn as sns
import math



'''

Version: cython friendly, connections and positions randomly placed

'''

# ---------------------- BINOMIAL FUNCTION ------------------------

def Binomial_fun(n,p, _vectorisation_idx):
    '''Generate a number from an exponential distribution using inverse
       transform sampling'''
    uniform = np.random.rand(n)
    return sum(uniform < p)




Binomial_fun = Function(Binomial_fun, arg_units=[1,1], return_unit=1,
                            stateless=False, auto_vectorise=True
                            )

cython_code = '''
 

cdef double Binomial_fun(int n,double p,_vectorisation_idx):

    cdef int count = 0
    cdef double uniform
    cdef int i
  
    
    for i in range(n):
        uniform=rand(_vectorisation_idx)
        
        if uniform < p:
            count = count+1
            
    return count;

'''
Binomial_fun.implementations.add_implementation('cython', cython_code,
                                                    dependencies={'rand': DEFAULT_FUNCTIONS['rand']})





def plot_layered_connections_with_mea_planar(neurons, astrocytes, gj_synapses, neuron_synapses, Grid):
    """
    Plots the positions of neurons, astrocytes, and MEA electrodes (as planar discs)
    in separate Z-layers and visualizes connections in 3D.

    Args:
        neurons (NeuronGroup): The neuron population. Assumed to have .x_neuron, .y_neuron.
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
    neuron_x = neurons.x_neuron / scale_factor
    neuron_y = neurons.y_neuron / scale_factor
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
    Optimized function to create a distance-based adjacency matrix.

    The ADJ matrix must comply with the following convention:
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
    x_Siff = coords[:, 0][:, np.newaxis] - coords[:, 0]
    y_diff = coords[:, 1][:, np.newaxis] - coords[:, 1]
    distances = np.sqrt(x_Siff**2 + y_diff**2)

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
def Get_norm(tr,td):
    
    '''
    This function calculates the Syanptic amplitude scaling factor. It is used to 
    scale the post-synaptic currents amplitude
    
    Params:
        td = decay time scale
        tr = rise time scale
       
    '''

    rise_ratio = tr / (td - tr)
    decay_ratio = td / (td - tr)
    Numerator = 1
    Denominator = ((tr / td) ** decay_ratio) - ((tr / td) ** rise_ratio)
    
    Norm = Numerator / Denominator


    return Norm


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
        't_off' : Inf*second,      # End of astrocyte stimulation (used in standalone gliotransmission)
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
        # I_bias (see below)       # IP_3 bias
        # --- Agonist-dependent IP_3 production
        'O_beta': 1.*umole/second,  # Maximal rate of IP_3 production by PLCbeta
        'O_N': 0.3/umole/second,   # Agonist binding rate
        'Omega_N': 1.8/second,     # Inactivation rate of GPCR signalling
        'K_KC': 0.5*umole,         # Ca^2+ affinity of PKC
        'zeta': 2.,                # Maximal reduction of receptor affinity by PKC
        'n': 1.,                   # Cooperativity of agonist binding reaction
        # --- Gliotransmitter release and time course        
        'C_Theta': 0.3*umole,      # Ca^2+ threshold for exocytosis
        'Omega_A': 0.6/second,     # Gliotransmitter recycling rate
        'U_A': 0.6,                # Gliotransmitter release probability
        'G_T': 200.*mmole,         # Total vesicular gliotransmitter
        'rho_e': 6.5e-4,           # Ratio of astrocytic vesicle volume/ESS volume
        'Omega_e': 5./second,      # Gliotransmitter clearance rate (think about distributed release)
        'spill_over': 0.75,         # Spill over parameter
        
        # Connection probability
        'conn_dist' : 200, # [um] 
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
    
    
    
    Neuron_area =  300*umetre**2
    
    
    params = { # --- Neuron Parameters
    'area': Neuron_area,               # membrane area of the neuron
    'Cm': (2*ufarad*cm**-2) * Neuron_area, # membrane capacitance (calculated with area)
    'El': -39.2 * mV,                    # Nernst potential of leaky ions
    'EK': -80 * mV,                      # Nernst potential of potassium
    'ENa': 70 * mV,                      # Nernst potential of sodium
    'g_na': 1.6 * 50 * msiemens * cm**-2 * Neuron_area, # maximal conductance of sodium channels (calculated with area)
    'g_kd': 1.3 * 5 * msiemens * cm**-2 * Neuron_area,  # maximal conductance of potassium (calculated with area)
    'gl': (0.3*msiemens*cm**-2) * Neuron_area, # maximal leak conductance (calculated with area)
    'VT': -30.4*mV,                      # alters firing threshold of neurons
    'sigma': 4 * mV,      #4.1               # standard deviation of the noisy voltage fluctuations #!!!
    
    # AHP current
    'g_AHP' : 5 * nS,                      # Maximum conductance of sAHP channels
    'tau_Ca' : 8000 * ms ,                 # recovery time constant sAHP channels
    'alpha_Ca' : 0.00035,                  # strength of the spike-frequency adaptation

    'I_inj': 15*pA, # Injected current # 18
 
     # Synaptic contribution
     'delta' : 0.6, # changes NMDAR/AMPAR ratio, should be between -1 and 1
     'E_ampa': 0 * mV,
     'E_nmda': 0 * mV,
     
     
     # Position
     'c_min' : 0, #[um]
     'c_max' : 1100, # [um]
    
     }
    
    
    params.update(kwargs)
    

    
    params.update({

     'g_ampa': (1 + params['delta']) * nS, # maximal conductance of AMPA channels
     'g_nmda': (1 - params['delta']) * nS, # maximal conductance of NMDA channels
        
        
        })
    

    return params






def get_Synparam(synapse_type='depressing',**kwargs):
    
    params = {
        
        'area' : 300*umetre**2,  
        # --- Synaptic dynamics
        'E_Iper': 80/100,          # Persentage of excitatory connections
        # Omega_d (see below)      # Depression rate
        # Omega_f (see below)      # Facilitation rate,
        # U_0_sr (see below)    # Basal synaptic release probability
        'Omega_c': 40./second,     # Neurotransmitter clearance rate
        'rho': 0.005,            # synaptic vesicle-to-extracellular space volume ratio
        'Y_T': 500.*mmole,         # Total neurotransmitter synaptic resource (in terms of vesicular concentration)
        # --- Presynaptic receptors
        'O_G': 1.5/umole/second,   # Agonist binding rate (activating)
        'Omega_G': 0.5/(60*second),# Agonist release rate (inactivating)
        # alpha (see below)        # Gliotransmitter effect on synaptic release
        # --- SIC/SOC
        'G_sic'     : 4.5*mV,      # Max SIC/SOC depolarization
        'tau_sic_r' : 30.*ms,      # SIC/SOC rise time constant
        'tau_sic' : 600.*ms,       # SIC/SOC decay time constant
        
       # Neurotransmitter release time constants
       'tau_rise_NT': 1*ms,
       'tau_decay_NT': 25*ms, # 
        
       
    
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
        
        
        'epsilon': 1e-40 * Hz,
        
        # Params of the kinetic model post-syn
        'tau_rise_ampa': 1*ms,
        'tau_decay_ampa': 2*ms,
        'tau_rise_nmda': 2*ms,
        'tau_decay_nmda': 100*ms,
        
        # Synaptic efficacy (function of the [GLU] in the cleft)
        'Xi_ampa':  0.5/mmole,
        'Xi_nmda': 0.3/mmole,
        
        # Connection probability
        'conn_prob' : 0.107, # Random 
        
        # Distance dependent
        'slope': 1/500, # in [um] sHOULD BE 500
        'intercept': 0.8, # 1 usually
        
        # Connection to astrocyte rules
        'Conn_syn_astro_cutoff' :70 *um,
        'sigma_A': 150*um,   #150*um,

       
    
       # Asynchronous Release parameters (uncommented and added to dictionary)
 
       'Omega_f_ar': 1/ (0.7 * second),
       'Uar': 0.001, #0.003
       'Umax': 0.5/ms,
       'x0': 0.2, # x0 seems to be unitless here
    
    }
    


    # ------------------ SYNAPSES ------------------
    if synapse_type == 'depressing':
        params.update({
            'Omega_d': 2./second,
            'Omega_f_sr': 3.33/second,
            'U_0_sr': 0.6,
            'alpha_syn': 0.,
        })
    elif synapse_type == 'facilitating':
        params.update({
            'Omega_d': 2./second, #2
            'Omega_f_sr': 2./second,
            'U_0_sr': 0.15,
            'alpha_syn': 1.       #1.,  
        })
    elif synapse_type == 'neutral':
        params.update({
            'Omega_d': 3./second,
            'Omega_f_sr': 3./second,
            'U_0_sr': 0.5,
            'alpha_syn': 1.,
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
        'tau_ca': 20.0*ms, # Intrasynaptic Ca2+ decay constant
        'Cpre'  : 1.0,     # Presynaptic Ca2+ increase per spk
        'Cpost' : 2.0,     # Postynaptic Ca2+ increase per spk
        'Theta_d' : 1.0,   # LTD threshold
        'Theta_p': 1.3,    # LTP threshold
        'gamma_d': 200.0,  # LTD learning rate
        'gamma_p': 321.808,# LTP learning rate
        'W_0'    : 0.5,    # LTP/LTD boundary
        'tau_w'  : 346.3615*second, # Time decay of synaptic weights
        'D'      : 13.7*ms,# Synaptic delay
        'sigma_'  : 2.8284, # variance in the diffusion approx,
        'beta'   : 0.5,
        'b'      : 5.
    })
    

    
    params.update(kwargs)

    return params





def Neuronal_Network(Nn,Syn_pdist = None,ics = False, Simulated_network = 'Neuronal',
                     Decay_type = 'Double_exp',synapse_type = 'neutral',conn_prob_ = None, seed_neu=None,seed_syn = None):
    

# Syn_pdist: stores statistical informations used to spatially reallocate the synapses. [0] probability of a connection to 
#   be set at the distance in [1]   
    
# ---------------------- NEURONAL GROUP ----------------------
     # neuron model
    eqs_NN = Equations('''
    
    dV/dt = noise + (-gl*(V-El)-g_na*(m*m*m)*h*(V-ENa)-g_kd*(n*n*n*n)*(V-EK)+ I_AHP +I-I_syn)/Cm  : volt
    dm/dt = alpha_m*(1-m)-beta_m*m : 1
    dh/dt = alpha_h*(1-h)-beta_h*h : 1
    dn/dt = (alpha_n*(1-n)-beta_n*n) : 1

    alpha_m = 0.32*(mV**-1)*4*mV/exprel((13*mV-V+VT)/(4*mV))/ms : Hz
    beta_m = 0.28*(mV**-1)*5*mV/exprel((V-VT-40*mV)/(5*mV))/ms : Hz
    alpha_h = 0.128*exp((17*mV-V+VT)/(18*mV))/ms : Hz
    beta_h = 4./(1+exp((40*mV-V+VT)/(5*mV)))/ms : Hz
    alpha_n = 0.032*(mV**-1)*5*mV/exprel((15*mV-V+VT)/(5*mV))/ms : Hz
    beta_n = .5*exp((10*mV-V+VT)/(40*mV))/ms : Hz

    I_AHP = -g_AHP*Ca*(V-EK) : amp
    dCa/dt = - Ca / tau_Ca : 1 

    
    noise = sigma*(2*gl/Cm)**.5*randn()/sqrt(dt) : volt/second (constant over dt)
    I : amp
    I_cell = -gl*(V-El)-g_na*(m*m*m)*h*(V-ENa)-g_kd*(n*n*n*n)*(V-EK): amp
    x_neuron : meter
    y_neuron : meter
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
        
          # r_Ar and r_Sr refer to the released NT in asynchronous and synchronous processes.  
       
        
        eqs_Syn = Equations('''
        
   
            # Available neurotransmitter
            dx_S/dt = Omega_d * (1 - x_S) -  r_Ar: 1 (clock-driven)
            
            # Usage of releasable neurotransmitter per single action potential (synchronous):
            dusr/dt = -Omega_f_sr * usr : 1 (clock-driven)
            
            
            # Add the asyncronous release
            r_Ar = x0*nar : Hz
            nar = Binomial_fun(int(floor(x_S/x0)),uar*dt)/dt :Hz (constant over dt)
            duar/dt = -uar*Omega_f_ar :Hz (clock-driven)
            
            r_Sr : 1 
           
            # avail = int(floor(x_S/x0)) : 1
            
            # Astrocyte ID for connection
            astro_index : integer
            
            # Positions
            x_syn : metre
            y_syn : metre
          
            ''')
        
        # -------------- Event based update --------------
        
        
        pre = '''
        
            U_0 =  U_0_sr
            usr += U_0 * (1 - usr)
            r_Sr = usr * x_S # synchronously released synaptic neurotransmitter resources
            x_S -= r_Sr     
            uar += Uar*(Umax-uar)
            '''
        post = None
        
        
       
           
        
        
    else:
        
        eqs_Syn = Equations('''
            # Fraction of activated presynaptic receptors
            dGamma_S/dt = O_G * G_A_syn * (1 - Gamma_S) - Omega_G * Gamma_S : 1 (clock-driven)
    
            
            # Available neurotransmitter
            dx_S/dt = Omega_d *(1 - x_S) -  r_Ar: 1 (clock-driven)
            
            # Usage of releasable neurotransmitter per single action potential (synchronous):
            dusr/dt = -Omega_f_sr * usr : 1 (clock-driven)
            
            
            # Add the asyncronous release
            r_Ar = x0*nar : Hz
            nar = Binomial_fun(int(floor(x_S/x0)),uar*dt)/dt :Hz (constant over dt)
            duar/dt = -uar*Omega_f_ar :Hz (clock-driven)
           
            
            # Define the variables of the model
            G_A_syn : mole  # gliotransmitter concentration in the extracellular space
            r_Sr : 1 
            
         
            # Astrocyte ID for connection
            astro_index : integer
            # Per-synapse gliotransmitter-effect parameter
            # alpha  : 1
            
            # Positions
            x_syn : metre
            y_syn : metre
            ''')
        
        # -------------- Event based update --------------
    
        pre = '''
        
            U_0 =  (1 - Gamma_S) * U_0_sr + alpha_syn * Gamma_S
            
            usr += U_0 * (1 - usr)
            r_Sr = usr * x_S # synchronously released synaptic neurotransmitter resources
            x_S -= r_Sr     
            uar += Uar*(Umax-uar)
            
        '''
        post = None
        
        
      
         
        
        
    # ---------- Extrasyn glutamate model ----------
    
    if Decay_type == 'Single_exp':
        
        
        eqs_Syn += Equations('''
                             
                             dY_S/dt = -Omega_c * Y_S + rho * Y_T * r_Ar  : mole (clock-driven)
                             
                             ''')
        
        pre +=  '''
        
                Y_S += rho * Y_T * r_Sr
        
                ''' 

    if Decay_type == 'Double_exp':
        
        
        eqs_Syn += Equations('''
                             
                             
                             dY_S/dt = ((tau_decay_NT / tau_rise_NT) ** (tau_rise_NT / (tau_decay_NT - tau_rise_NT))*x_Y_S-Y_S)/tau_rise_NT : mole (clock-driven)
                             dx_Y_S/dt = -x_Y_S/tau_decay_NT +  rho * Y_T * r_Ar                                               : mole (clock-driven)
                             
                            
                             
                             ''')
        
        
        pre +=  '''
        
                x_Y_S += rho * Y_T * r_Sr
        
                ''' 

    
    # ----------- SYNAPTIC CURRENTS MODEL -------------
    
    
    eqs_Syn += Equations('''
                         
                               
                            dr_ampa/dt = -r_ampa/tau_decay_ampa + (rho * Y_T * r_Ar * Xi_ampa) : 1 (clock-driven)
                            
                            dr_nmda/dt = ((tau_decay_nmda / tau_rise_nmda) ** (tau_rise_nmda / (tau_decay_nmda - tau_rise_nmda))*x_r_nmda-r_nmda)/tau_rise_nmda  : 1 (clock-driven)
                            dx_r_nmda/dt = -x_r_nmda/tau_decay_nmda +  (rho * Y_T * r_Ar * Xi_nmda)  : 1 (clock-driven)
                            
                           
                            r_ampa_tot_post = r_ampa : 1 (summed)
                            r_nmda_tot_post = r_nmda : 1 (summed)
                         
                         
                        ''')
                        
                        
    pre += '''           
           r_ampa +=   (rho * Y_T * r_Sr * Xi_ampa)
           x_r_nmda += (rho * Y_T * r_Sr * Xi_nmda) 
           
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
    
    
    N = NeuronGroup(Nn, model=eqs_NN, name='Neuron*',namespace= params_NN, threshold='V>20*mV',  reset='Ca += alpha_Ca',refractory=2 * ms,
                        method='exponential_euler',dtype=float32)
    
    # Initialize neuron parameters
    N.V = -39 * mV                          # approximately resting membrane potential
    
    
    
    
    # ----- SET POSITIONS AND CONNECTIONS -----
    # Position neurons on a grid

    Coordinates = get2D_rnd_coordinates(N.N,params_NN['c_min'],params_NN['c_max'],seed_neu)
    N.x_neuron = Coordinates[:,0]*um
    N.y_neuron = Coordinates[:,1]*um
    
    
    
    
    S = Synapses(N,N, model=eqs_Syn,
                        on_pre=pre,
                        on_post=post,
                        name='Synapse*',
                        namespace=params_Syn,
                        method='exponential_euler',dtype=float32,
                        )
    
    S.namespace['Binomial_fun'] = Binomial_fun
    
    # -------------- Connections --------------
    
    # --- Random ---
    S.connect(p=params_Syn['conn_prob'],condition='i != j',)
    
       
    
    
    
    # Set synapse position coincident with the post-synaptic neuron plus a stochastic shift (See notes)

    
    
    Syn_coordinates =  get_synapse_coordinates(S,N,
                                               Syn_pdist['Syn_prob'].to_numpy(),
                                               Syn_pdist['Radius_val'].to_numpy()
                                               )
    Syn_coordinates = np.vstack(Syn_coordinates)

    # Print the shape (number of rows, number of columns)

    
    S.x_syn = Syn_coordinates[:, 0] * um
    S.y_syn = Syn_coordinates[:, 1] * um
    
   
        
    # S Initialization --------------
    S.x_S = 1.0
    
    
    
    # Random initialization of initial conditions
    if ics=='rand':
        S.u_S = 'rand()'
        S.x_S = 'rand()'
        Y_T = params['Y_T']
        S.Y_S = '1.2 * rho_c * Y_T * rand()'
        
    return N,S
        
    



# -------------- ASTROCYTE GROUP --------------
def astrocyte_connections(Astrocyte_group,Connection_dist):
    
    '''
    This function aims to define pre and post astrocyte for connections
    '''
    Na = Astrocyte_group.N
    
    # Generate the KDTree form the neuronal position data
    x_pos = np.array(Astrocyte_group[:].x_astro/um)
    y_pos = np.array(Astrocyte_group[:].y_astro/um)
    
    pos = np.column_stack((x_pos, y_pos))

    # Generate the KDTree
    Astro_positions = KDTree(pos)
    
    Source = []
    Target = []
    
    astro_idx = 0
    for astro in pos:
        
        # Extract indicies
        A_idx = Astro_positions.query_radius(astro.reshape(1, -1), r=Connection_dist)
        A_idx = np.array(A_idx[0])
        
        if A_idx.size == 0: # Empty array....no post units
            continue
        
        # Construct pre vector
        source_astro = np.full(len(A_idx), astro_idx)
        
        # Update variables 
        Source.append(source_astro)
        Target.append(A_idx)
        
        astro_idx = astro_idx+1
        
        
        
    
        
    return np.concatenate(Source),np.concatenate(Target)  
    
    
    
    
    

def Astrocyte_Group(N_astro,Simulated_network,seed_astro = None,ics =None):
# ------ Astrocyte core equations ------

    eqs_A = Equations('''
        # Fraction of activated astrocyte receptors:
        dGamma_A/dt = O_N * (Y_bias+Y_extra*spill_over)**n * (1 - Gamma_A) -
                      Omega_N*(1 + zeta * C/(C + K_KC)) * Gamma_A : 1 
    
        # IP_3 dynamics:
        dI/dt = O_beta * Gamma_A + O_delta/(1 + I/K_delta) * C**2/(C**2 + K_delta**2) -
                O_3K * C**4/(C**4 + K_D**4) * I/(I + K_3K) - Omega_5P*I +
                I_coupling_tot : mole 
    
      
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
        ''')
       
       
       
    Params_astroGT = get_Astroparam()
       
    # The definition of a threshold and reset mechanism in the astrocyte group
    # allows to use SpikeMonitors to estimate the frequency of oscillations
    Astro = NeuronGroup(N_astro, eqs_A,
                        threshold='C>C_osc',
                        refractory='C>C_osc',
                        method='gsl_rkf45',
                        namespace=Params_astroGT,
                        name='astrocyte*',dtype=float32)
    
    # Random initialization of initial conditions
    if ics=='rand':
        Astro.Gamma_A = 'rand()'
        Astro.I = '3*rand()*umole'
        Astro.C = '1.5*rand()*umole'
        Astro.h = 'rand()'
        
        
    # ----- SET POSITIONS -----
    # Position neurons on a grid
    Coordinates = get2D_rnd_coordinates(N_astro,Params_astroGT['c_min'],Params_astroGT['c_max'],seed_astro)
    Astro.x_astro = Coordinates[:,0]*um
    Astro.y_astro = Coordinates[:,1]*um
        
        
    # ----- Gap-junction based astro links -----
    
    Gap_Eq = Equations('''
                     
            
                        delta_I = I_post - I_pre: mole
                        I_coupling = -F/2*(1 + tanh((abs(delta_I) - I_Theta)/omega_I))*sign(delta_I) : mole/second 
                        I_coupling_tot_post = I_coupling : mole/second (summed)
                        
                       
                        ''')
    
    GJ = Synapses(Astro,Astro,
                  model=Gap_Eq,
                  method='rk4',
                  namespace= Params_astroGT,
                  name = 'Gap_junctions*',dtype=float32
                  )
    
    
    
    
    # ----- Connections -----
    '''
    Astros are connected by gap-junctions within distance of 100 um
    
    Paper: A Computational Model of Interactions Between Neuronal and 
        Astrocytic Networks: The Role of Astrocytes in the Stability of the Neuronal Firing Rate
    
    '''

    # --------- RANDOM -----------
    Source,Target = astrocyte_connections(Astro,Params_astroGT['conn_dist'])
    GJ.connect(i=Source , j= Target)

   
    
    
    
    
    if Simulated_network == 'Astrocytic':
        
        import random
        random.seed(sed)    
        
        
    # ---------- EXTERNAL STIMULATION ----------
        Params_astroGT.update({'tau_glustim' :25*ms}) # As for synapse params
        Params_astroGT.update({'Y_bias_max' : 1*mmole}) # Maximum glutamate concentration
        Params_astroGT.update({'poisson_rate' : 2*Hz}) # Lambda parameter of the poisson process
        Params_astroGT.update({'N_stim' : 40}) # Lambda parameter of the poisson process
        
        
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
def Gliotransmission(N_astro,Astro,ics = None):
    
    eqs_GT = Equations('''
            # Gliotransmitter
            C : mole (linked)
            dx_A/dt = Omega_A * (1 - x_A) : 1   # Fraction of gliotransmitter resources available for release
            dG_A/dt = -Omega_e*G_A  : mole  # gliotransmitter concentration in the extracellular space
            ''')
    gliot_release = '''
    G_A += rho_e * G_T * U_A * x_A
    x_A -= U_A *  x_A 
    '''
    threshold = 'C>C_Theta'
    refractory = 'C>C_Theta'
    
    Params_astroGT = get_Astroparam()
    Glio_release = NeuronGroup(N_astro, eqs_GT,
                            # The following formulation makes sure that a "spike" is
                            # only triggered at the first threshold crossing
                            threshold=threshold,
                            refractory=refractory,
                            # The gliotransmitter release happens when the threshold
                            # is crossed, in Brian terms it can therefore be
                            # considered a "reset"
                          
                            reset=gliot_release,
                            method='gsl',
                            name='gliot_release*',
                            namespace=Params_astroGT,dtype=float32)
    
    # Assign initial conditions
    Glio_release.x_A = 1
    Glio_release.G_A = 0.0*mole
    Glio_release.C = linked_var(Astro, 'C')
    
    # Random initialization of initial conditions
    if ics=='rand':
        synapses.x_A = 'rand()'
        synapses.G_A = '1.2 * rho_e * G_T * rand()'
        
    return Glio_release


# -------------- SYNAPSE-ASTRO LINK ---------------


    
def Synapse_to_astro(synapse,Astro):
    # ---- Syn-astro ----

    Syn_Astro = Synapses(synapse,Astro,
                        model='''
                        # neurotransmitter concentration in the extracellular space
                        Y_extra_post = Y_S_pre : mole (summed)
                        ''',
                        namespace=synapse.namespace,
                        method = 'gsl ',
             
                        
                        name="ecs_syn_to_astro*")
    
    
        
        # --------- RANDOM -----------

    p_conn = 'exp(- ((sqrt((x_syn_pre - x_astro_post)**2 + (y_syn_pre - y_astro_post)**2))**2) / (2 * sigma_A**2))'
    Syn_Astro.connect(
    condition='sqrt((x_syn_pre - x_astro_post)**2 + (y_syn_pre - y_astro_post)**2) < Conn_syn_astro_cutoff' ,
    p=p_conn
    )
    
    Source = list(Syn_Astro.i) # Synapse as pre unit
    Target = list(Syn_Astro.j) # Astro as target unit
    

    return Syn_Astro,Source,Target # To be loaded in Astro_to_syn swapped.



def Astro_to_Syn(Glio_release,synapse,Source_astro, Target_syn):
    
    '''
    The connectivity is bidirectional thus the same as for the function 'Synapse_to_astro'
    from which the Sources and target are inherted. Be aware that the source in this case is the target
    of the provious unction
    
    
    '''
    
    # ---- Astro-syn ----
    # Glio_relaease is the reference neuronal group
    Astro_Syn = Synapses(Glio_release,synapse,
                             model='''
                             # gliotransmitter concentration in the extracellular space
                             G_A_syn_post = G_A_pre : mole (summed)
                             ''',
                             method = 'gsl ',
                  
                             name="ecs_astro_to_syn*",dtype=float32
                             )
    
    # ---- Connections ----

    
    Astro_Syn.connect(i = Source_astro, j = Target_syn)
    
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
        
def sample_distance_from_soma(syn_prob, rarius_val):
    '''
    Samples a single distance from the soma based on the provided probability distribution.

    Args:
        syn_prob (np.array): Array of connection probabilities for each shell.
        rarius_val (np.array): Distances from the soma corresponding to the shells.

    Returns:
        float: A single distance value (in um) sampled from the distribution.
    '''
    
    # Set the seed
    rng_syn =  np.random.default_rng()
    
    # Normalize the probabilities to ensure they sum to 1
    prob_sum = np.sum(syn_prob)
    if prob_sum == 0:
        # Fallback to a uniform distribution if probabilities are all zero
        normalized_probs = None
    else:
        normalized_probs = syn_prob / prob_sum
        
    # Use np.random.choice to sample a distance based on the probabilities
    # We choose from the radial values where the probability is non-zero
    valid_indices = np.where(normalized_probs > 0)[0]
    
    if len(valid_indices) == 0:
        print("Warning: All probabilities are zero. Falling back to uniform sampling.")
        return rng_syn.choice(rarius_val)
    
    sampled_distance = rng_syn.choice(rarius_val[valid_indices], p=normalized_probs[valid_indices],replace = True)
    
    return sampled_distance   
def get_synapse_coordinates(Synapse,Neuron,Syn_prob,rarius_val,displ_bias=15):
    """
    Calculates the 2D coordinates of each synapse based on a distance from the postsynaptic neuron.

    This function iterates through each synapse, calculates the vector from the presynaptic
    to the postsynaptic neuron, and then determines the synapse's position by moving
    a sampled distance along that vector from the postsynaptic neuron's position.

    No autaptic connections allowed
    A slight bias of 10um is introduced
    
    EVERYTRHING is expressed in um

    Args:
        Synapse (brian2.Synapses): The Synapses object containing the connections.
        Neuron (brian2.NeuronGroup): The presynaptic/postsynaptic neuron group.


    Returns:
        list: A list of (x, y) tuples, where each tuple is the coordinate of a synapse.
    """
    # Initialize an empty list to store the synapse coordinates
    synapse_coords = []
    distance_ = []
    # Iterate through each synapse
    for i in range(len(Synapse)):
        # Get the indices of the pre- and postsynaptic neurons for the current synapse
        pre_idx = Synapse.i[i]
        post_idx = Synapse.j[i]

        # Get the coordinates of the pre- and postsynaptic neurons
        pre_pos = np.array([Neuron.x_neuron[pre_idx]/um, Neuron.y_neuron[pre_idx]/um])
        post_pos = np.array([Neuron.x_neuron[post_idx]/um, Neuron.y_neuron[post_idx]/um])

        # Calculate the vector from the postsynaptic to the presynaptic neuron
        vector = post_pos - pre_pos

        # Calculate the length (magnitude) of the vector
        vector_length = np.linalg.norm(vector)

        # Normalize the vector to get a unit vector
        unit_vector = vector / vector_length

        # Sample a distance for this synapse
        # CHeck that the synapse distance is lower than then the neuron's.
        
        distance = sample_distance_from_soma(Syn_prob,rarius_val) + displ_bias
        
        while distance > vector_length:
            distance = sample_distance_from_soma(Syn_prob,rarius_val) + displ_bias
        
        distance_.append(distance)

        # Calculate the new coordinate by moving 'distance' along the unit vector from the postsynaptic neuron
        new_coord = post_pos - unit_vector * distance

        # Append the new coordinate as a tuple to the list
        synapse_coords.append(tuple(new_coord))

    return synapse_coords
