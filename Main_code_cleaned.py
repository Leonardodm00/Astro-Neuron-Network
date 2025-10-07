



import matplotlib.pyplot as plt
from brian2 import *

from brian2 import clear_cache

import pandas as pd
import os


os.chdir(r'C:\Users\Admin\Desktop\Leonardo\ASN')
from ASN_fun_BD import *
#%
'''
Version: cython friendly, connections and positions randomly placed

The phantom network is defined as follows:
    
    
         S          S          S   
    N -------> N -------> N -------> N 
        \\         \\         \\
        \\         \\         \\
        A <------> A <-------> A



N = neuron;
S = synapse;
A = astrocyte;
\\ = Bidirectional vescicle exocitosys-driven connections between astrocytes and synapses;
<--> = Gap-junction mediated links between astrocytes;
--> = Synaptic connections.

FEATURES:
    1) Neurons and astrocytic positions are randomly set.
    2) Only the TM_coupled model is used for synaptic weight modulation
    3) Asynchronous release targets both ampa and nmda receptors
    4) The synaptic model scales r_nmda and r_ampa by an alpha factor to match the currents observed
    5) Autapses are not modeled.
    6) For Glia-Syn connections the rule is distance based as in [2]. The synapse position is located at the post-synaptic 
        neuron given the discrepances in the extention of axonal and dendritic domains. THIS DOEN'T WORK, indeed given that an astrocyte 
        connection is based on th erelative position of the astro itself and the synapse, I end up with a (unfeasably) huge amount of synapses
        linked to the same astrocyte. FOr this reason a slight shift is added from the location of the post-syn neuron to set the synapse coordinates.
        To define the distance dependend probability that a synapse is located at a certain distance from the post-syn neuron we focus on
         the morphometric analysis on hipsc control lines. A better explanation is given in the function 'get_dendrite_prob' in ASN_fun script.
        Finally given the distance from the soma the coordinate of the synapse will be define by the soma location shifted about the defined distance
        along the axes linking the pre and post syn neurons.



IMPORTANT NOTE

When using standalone C++ code the variables are not initialized sequentilly as when using cython. Instead the whole code is 
set. This leads to problems when one group state variable depends on one defined before. This holds for example for synapse positions
that retrieve neuronal's pnes. A different approach is needed. A way is to initialize the code in cython and save all the neccessary 
variables, such as the synapse position.


 Cannot retrieve the values of state variables in standalone code before the simulation has been run.    



REFERENCES:
    1) Distance dependent prob of conneciton = Estimating neuronal connectivity from axonal and dendritic density fields
    2) Astrocyte connections: A Computational Model of Interactions Between Neuronal and Astrocytic Networks: The Role of Astrocytes in the Stability of the Neuronal Firing Rate

    
    
    
TODO:
    1) Syn to astro and astro to syn functions must be improved by defining a distance dependent connections.
    2) Add ics in the astrocytes
'''


#%



def exponential_rand(n,p, _vectorisation_idx):
    '''Generate a number from an exponential distribution using inverse
       transform sampling'''
    uniform = np.random.rand(n)
    return sum(uniform < p)

exponential_rand = Function(exponential_rand, arg_units=[1,1], return_unit=1,
                            stateless=False, auto_vectorise=True
                            )

cython_code = '''
 

cdef double exponential_rand(int n,double p,_vectorisation_idx):

    cdef int count = 0
    cdef double uniform
    cdef int i
  
    
    for i in range(n):
        uniform=rand(_vectorisation_idx)
        
        if uniform < p:
            count = count+1
            
    return count;

'''
exponential_rand.implementations.add_implementation('cython', cython_code,
                                                    dependencies={'rand': DEFAULT_FUNCTIONS['rand']})



# Saving 
Out_path = r'C:\Users\Admin\Desktop\Leonardo\ASN\Output Temp'


BrianLogger.suppress_hierarchy('brian2.devices')
BrianLogger.suppress_hierarchy('brian2.parsing')


# Clear the cache for the 'cython' code generation target
start_scope()
 # ------------------------- SET OPTIONS -------------------------

# ------- Synapses -------

# 1. Define the filename
os.chdir(r'C:\Users\Admin\Desktop\Leonardo\ASN')
filename = 'synapse_pdist.csv'
# 2. Load the file
Syn_pdist = pd.read_csv(filename)

Syn_Currents_model = 'TM-coupled' # 'Kinetic','Nina','TM-coupled'


# ------------------------- PARAMETERS -------------------------

# --------- SIMULATION -----------
simtime =15 * second               # simulation time
# transient = 3 * second  
seed_device = 50            # time omitted as transient
seed_neuron = 39                             # random number seed
seed_synapse = 35
seed_astro= 60
devices.device.seed(seed_device)            # set the seed for all the random number realisations
defaultclock.dt = 0.05*ms

Simulated_network = 'Full' # Astrocytic/Neuronal/Full


# --------- NEURON -----------
Nn = 100
neuron_radius = 9 #[um]
# --------- SYNAPTIC -----------

# ----- Connectivity -----

conn_prob_ = 0.13

# --------- ASTROCYTE and GJ ----------- 
'''
Given the nature of the link hte adjency matrix is ALWAYS symmetric

'''
Na = 43


# --------- ELECTRODE RECORDINGS ----------- 
pitch = 300 #[um] 
electrode_radius = 15 #[um]

pitch_recsites = 7.5 # [um]  
shift = 11.25 # [um]

electrode_dist = 300 # [um]

c_min = 0 #[um]
c_max = 1100 #[um]


# ------------------------- GROUPS BUILD-UP -------------------------

if Simulated_network == 'Full':
    # --------- NEURON and SYNAPSE -----------
    N,S = Neuronal_Network(Nn,Syn_pdist = Syn_pdist,ics = False, Simulated_network = Simulated_network,
                         Decay_type = 'Double_exp',synapse_type = 'facilitating', conn_prob_ = conn_prob_,seed_neu=seed_neuron,seed_syn=seed_synapse)
    
    

    N.I = '(rand() -0.5) * I_inj'          # Make neurons heterogeneously excitable
    
    
    
    
    
    # --------- ASTROCYTE -----------
    Astro,GJ = Astrocyte_Group(Na,Simulated_network,seed_astro = seed_astro)
    
    
    
    
    # --------- GLIOTRANSMISSION ----------- 
    GT = Gliotransmission(Na,Astro)
    # GT.namespace['U_A'] = 0.*mmole
    
    
    # --------- ASTRO-NEURON LINKS ----------- 
    # --- Synapse to astro ---
    StoA,Source_syn,Target_astro  = Synapse_to_astro(S,Astro)
    
    # --- Astro to synapse ---
    AtoS = Astro_to_Syn(GT,S,Target_astro,Source_syn)   # Astro_to_Syn(Glio_release,synapse,Source_astro, Target_syn)
    
    
    
    
elif Simulated_network == 'Neuronal':

    # --------- NEURON and SYNAPSE -----------
    N,S = Neuronal_Network(Nn,Syn_pdist = Syn_pdist,ics = False, Simulated_network = Simulated_network,
                         Decay_type = 'Double_exp',synapse_type = 'facilitating', conn_prob_ = conn_prob_,seed_neu=seed_neuron,seed_syn=seed_synapse)
    
    
    # N.namespace['sigma']=4.1*mV
    # N.namespace['I_inj']=0*pA
    N.I = '(rand() -0.5) * I_inj'          # Make neurons heterogeneously excitable
   
    

    
    
    
elif Simulated_network == 'Astrocytic':   
 
    '''
    Glutamate stimulation is delivered to randomly choosen astrocytes in number 
    N_stim in the parameters.
    '''
    # --------- ASTROCYTE -----------
    Astro, GJ,P,Glu_Input = Astrocyte_Group(Na,Simulated_network,seed_astro = seed_astro)
    



# ------------------------- NETWORK SIMULATION -------------------------

# --- Monitors ---
# recording_stringN = ['V','I_syn','I_ampa','I_nmda','I_cell']
recording_stringN = ['V']
# recording_stringS = ['Y_S','r_Ar','nar']
# recording_stringA = ['C','I','Gamma_A','I_coupling_tot','Y_extra']
# recording_stringGT = ['G_A','x_A']


if Simulated_network == 'Full':
    
    # MonitorS = StateMonitor(S, recording_stringS, record=True)
    # MonitorA = StateMonitor(Astro, recording_stringA, record=True)
    MonitorN1 = StateMonitor(N, recording_stringN, record=True)
    # MonitorGT = StateMonitor(GT, recording_stringGT, record=True)
    SpikesN2 = SpikeMonitor(N)
    SpikesA2 = SpikeMonitor(Astro)
    

elif Simulated_network == 'Neuronal':
    MonitorS = StateMonitor(S, recording_stringS, record=True)
    MonitorN = StateMonitor(N, recording_stringN, record=True)
    SpikesN = SpikeMonitor(N)
    
    
elif Simulated_network == 'Astrocytic': 
    MonitorG = StateMonitor(Glu_Input, ['Y_bias_in'], record=True)
    MonitorA = StateMonitor(Astro, recording_stringA, record=True)
    SpikesA = SpikeMonitor(Astro)
    SpikesP = SpikeMonitor(P)
    
#%
# # %matplotlib
# plot_connections(N, Astro, S, StoA)
# --- Collect and add monitors ---
#%
net_ = Network(collect())  # automatically include all the stated groups
# BUILD
# for param in param_list:
#   device.rum(run_args={param})
net_.run(simtime,report='text', profile=True)

#%%
%matplotlib
plt.figure()
plt.plot(SpikesN2.t / second, SpikesN2.i, '.k', ms=4)
plt.show()



#%%
# Convert coordinates for plotting
neuron_x = N.x_neuron 
neuron_y = N.y_neuron 
synapse_x = S.x_syn
synapse_y = S.y_syn
pre_x = [N.x_neuron[i]  for i in S.i]
pre_y = [N.y_neuron[i]  for i in S.i]
post_x = [N.x_neuron[j]  for j in S.j]
post_y = [N.y_neuron[j]  for j in S.j]

# --- Plotting ---
plt.figure(figsize=(10, 10))

# Plot neurons
plt.scatter(neuron_x, neuron_y, s=150, c='blue', label='Neurons (Somas)', zorder=2)

# Plot synapses and lines from pre to post-synaptic neurons
for i in range(len(S.i)):
# for i in range(1):
    # Plot the line from pre to post neuron
    plt.plot([pre_x[i], synapse_x[i]], [pre_y[i], synapse_y[i]], 'r--', alpha=0.3, zorder=1)
    
    # Plot the line from post neuron to synapse
    plt.plot([post_x[i], synapse_x[i]], [post_y[i], synapse_y[i]], 'g--', alpha=0.6, zorder=1)
    
    # Plot the synapse location
    plt.scatter(synapse_x[i], synapse_y[i], s=50, c='red', marker='x', zorder=3)

# Add labels and title
plt.title('Visualization of Synapse Coordinates')
plt.xlabel('X Coordinate ($ \mu m $)')
plt.ylabel('Y Coordinate ($ \mu m $)')
plt.legend(['Neuron Somata', 'pre-synaptic','post-synaptic','Synapse Location'])
plt.gca().set_aspect('equal', adjustable='box')
plt.grid(True)
plt.show()
#%%
os.chdir(r'C:\Users\Admin\Desktop\Leonardo\ASN')
from ASN_fun import *
Grid = Get_12grid(pitch)


#  
plot_layered_connections_with_mea_planar(N, Astro, GJ,S,Grid)
# --------------------- PLOTS ---------------------
#%%
# ------- NEURONS -------
%matplotlib
fig, (ax1, ax2, ax3,ax4) = plt.subplots(4, 1) # Added figsize for better viewing



ax1.plot(MonitorN1.t / second, MonitorN1[0].V / mV, 'k', linewidth=0.7)
ax1.set_ylabel('Voltage [mV]')

ax2.plot(MonitorN1.t / second, MonitorN1[1].V / mV, 'k', linewidth=0.7)
ax2.set_ylabel('Voltage [mV]')



ax3.plot(MonitorN1.t / second, MonitorN1[2].V / mV, 'k', linewidth=0.7)
ax3.set_ylabel('Voltage [mV]')



ax4.plot(MonitorN1.t / second, MonitorN1[3].V / mV, 'k', linewidth=0.7)
ax4.set_xlabel('Time [s]')
ax4.set_ylabel('Voltage [mV]')

#%%
# def check_nan_synapse_states(monitor_s, recording_vars_s):
#     """
#     Finds synapses where state variables have NaN values and reports the time
#     of the first NaN occurrence.

#     Args:
#         monitor_s (StateMonitor): The Brian2 StateMonitor object for the synapse group.
#         recording_vars_s (list): A list of state variable names to check for NaNs.
#     """
#     nan_synapses = set()
#     num_synapses = monitor_s.N

#     print("\n--- Identifying Synapses with NaN Values ---")
    
#     for syn_index in range(num_synapses):
#         print(f'Evaluated Synapse: {')
#         found_nan_in_syn = False
        
#         # Iterate through each state variable we are monitoring
#         for var_name in recording_vars_s:
#             state_trace = getattr(monitor_s, var_name)[syn_index]
#             nan_indices = np.where(np.isnan(state_trace))[0]

#             if nan_indices.size > 0:
#                 nan_synapses.add(syn_index)
                
#                 # Get the time of the first NaN for this specific synapse and variable
#                 first_nan_index = nan_indices[0]
#                 time_of_nan = monitor_s.t[first_nan_index]
#                 formatted_time = f"{time_of_nan/second:.3f} s"
#                 print(f"❌ Synapse {syn_index} failed in variable '{var_name}' at {formatted_time}.")
#                 found_nan_in_syn = True
#                 break # Move to the next synapse once a NaN is found

#     if not nan_synapses:
#         print("✅ No NaN values were found in any synapse's state variables. No further check needed.")
#     else:
#         print(f"\n--- {len(nan_synapses)} Synapse(s) with NaN values found. ---\n")

def check_nan_neuron_astro_link(monitor_n, N, S, AtoS):
    """
    1. Finds neurons where voltage has NaN values.
    2. Determines if the synapses targeting those neurons are involved in
       Astrocyte-to-Synapse (AtoS) modulation, and identifies the
       linked astrocyte.

    Args:
        monitor_n (StateMonitor): The Brian2 StateMonitor object for the neuronal group (N).
        N (NeuronGroup): The main neuron group.
        S (Synapses): The main synapse group (N to N connections).
        AtoS (Synapses): The Astro-to-Synapse modulation group.
    """
    # 1. --- Identify Neurons with NaN Values ---
    nan_neurons = set()
    num_neurons = 50

    print("\n--- Identifying Neurons with NaN Values ---")
    for neuron_index in range(num_neurons):
        voltage_trace = monitor_n.V[neuron_index]
        nan_indices = np.where(np.isnan(voltage_trace))[0]

        if nan_indices.size > 0:
            nan_neurons.add(neuron_index)
            # Get the time of the first NaN for this specific neuron
            first_nan_index_for_neuron = nan_indices[0]
            time_of_nan = monitor_n.t[first_nan_index_for_neuron]
            formatted_time = f"{time_of_nan/second:.3f} s"
            print(f"❌ Neuron {neuron_index} failed (NaN detected) at {formatted_time}.")

    if not nan_neurons:
        print("✅ No NaN values were found in any neuron's voltage. No further check needed.")
        return # Exit the function early

    print(f"\n--- {len(nan_neurons)} Neuron(s) with NaN values found. ---\n")

    # 2. --- Check Linkage to Astrocyte-Modulated Synapses ---
    print("\n--- Checking Astrocyte Involvement ---\n")
    
    # Get the indices of the synapses that are modulated by the astrocyte (AtoS)
    astro_modulated_synapse_indices = AtoS.j[:]
    # Get the pre-synaptic astrocyte indices for the AtoS synapses
    astrocyte_indices = AtoS.i[:]
    
    # Get the post-synaptic neuron indices for ALL synapses (S)
    synapse_post_indices = S.j[:]

    neurons_with_astro_modulated_input = set()

    # Check each failed neuron
    for nan_neuron in nan_neurons:
        is_astro_linked = False
        linked_astrocytes = set()
        
        # Find all synapses in S that target the current nan_neuron
        synapse_indices_to_nan_neuron = np.where(synapse_post_indices == nan_neuron)[0]
        
        # Check if any of these synapses are in the set of astrocyte-modulated synapses
        for syn_index in synapse_indices_to_nan_neuron:
            # Check if this synapse index is in the list of astro-modulated synapses
            if syn_index in astro_modulated_synapse_indices:
                is_astro_linked = True
                neurons_with_astro_modulated_input.add(nan_neuron)
                
                # Find the index of the synapse in the AtoS group
                atos_syn_index = np.where(astro_modulated_synapse_indices == syn_index)[0]
                if atos_syn_index.size > 0:
                    # Use that index to find the linked astrocyte
                    linked_astro_index = astrocyte_indices[atos_syn_index[0]]
                    linked_astrocytes.add(linked_astro_index)
        
        # 3. --- Report Result ---
        if is_astro_linked:
            linked_astrocytes_str = ', '.join(map(str, sorted(list(linked_astrocytes))))
            print(f"🔥 Neuron {nan_neuron} IS CONNECTED to Astrocyte(s): {linked_astrocytes_str}.")
        else:
            print(f"❓ Neuron {nan_neuron} is NOT connected to an Astrocyte-Modulated Synapse (Failure likely propagated).")


# Example Usage (requires the groups and monitor from your simulation)
# check_nan_neuron_astro_link(MonitorN, N, S, AtoS)

def check_nan_astrocyte_states(monitor_a, recording_vars_a):
    """
    Finds astrocytes where state variables have NaN values and reports the time
    of the first NaN occurrence.

    Args:
        monitor_a (StateMonitor): The Brian2 StateMonitor object for the astrocyte group.
        recording_vars_a (list): A list of state variable names to check for NaNs.
    """
    nan_astrocytes = set()
    num_astrocytes = 3

    print("\n--- Identifying Astrocytes with NaN Values ---")
    
    for astro_index in range(num_astrocytes):
        found_nan_in_astro = False
        
        # Iterate through each state variable we are monitoring
        for var_name in recording_vars_a:
            state_trace = getattr(monitor_a, var_name)[astro_index]
            nan_indices = np.where(np.isnan(state_trace))[0]

            if nan_indices.size > 0:
                nan_astrocytes.add(astro_index)
                
                # Get the time of the first NaN for this specific astrocyte and variable
                first_nan_index = nan_indices[0]
                time_of_nan = monitor_a.t[first_nan_index]
                formatted_time = f"{time_of_nan/second:.3f} s"
                print(f"❌ Astrocyte {astro_index} failed in variable '{var_name}' at {formatted_time}.")
                found_nan_in_astro = True
                break # Move to the next astrocyte once a NaN is found

    if not nan_astrocytes:
        print("✅ No NaN values were found in any astrocyte's state variables. No further check needed.")
    else:
        print(f"\n--- {len(nan_astrocytes)} Astrocyte(s) with NaN values found. ---\n")

check_nan_neuron_astro_link(MonitorN1, N, S, AtoS)
# check_nan_astrocyte_states(MonitorA, recording_stringA)
# check_nan_synapse_states(MonitorS2, recording_stringS,S)
#%%
def find_first_nan_index(Simulated_network, monitors):
    """
    Analyzes all state monitors to find the index (time step) where the 
    first NaN value appears across all recorded neurons, synapses, or astrocytes.

    Args:
        Simulated_network (str): The type of network simulated ('Full' or 'Neuronal').
        monitors (dict): A dictionary containing the StateMonitor objects (e.g., {'MonitorN': MonitorN, ...}).
    """
    
    # Define the variables to check for each group, based on your monitoring strings
    variable_map = {
        'MonitorN': ['V', 'I_syn', 'I_ampa', 'I_nmda', 'I_cell'],
        # Use the synapse monitor specific to the network type
        'SynapseMonitor': ['usr', 'x_S', 'Y_S', 'uar', 'r_Sr', 'r_Ar', 'r_ampa', 'r_nmda', 'Gamma_S'],
        'MonitorA': ['C', 'I', 'Gamma_A', 'I_coupling_tot', 'Y_extra'],
        'MonitorGT': ['G_A', 'x_A'],
        'MonitorG': ['Y_bias_in'] # Only used in 'Astrocytic'
    }

    nan_found = False
    
    # Map the correct synapse monitor based on the simulation type
    if Simulated_network == 'Full':
        syn_monitor_key = 'MonitorS2'
    elif Simulated_network == 'Neuronal':
        syn_monitor_key = 'MonitorS'
    else:
        # Handle 'Astrocytic' or other scenarios with a specific monitor list
        monitor_keys = ['MonitorA', 'MonitorG']
        if Simulated_network == 'Astrocytic':
            print(f"--- Searching for NaN in {Simulated_network} Network ---")
        else:
            print("Unknown network type. Checking available monitors.")


    monitor_keys = []
    if 'MonitorN' in monitors: monitor_keys.append('MonitorN')
    if syn_monitor_key in monitors: monitor_keys.append(syn_monitor_key)
    if 'MonitorA' in monitors: monitor_keys.append('MonitorA')
    if 'MonitorGT' in monitors: monitor_keys.append('MonitorGT')
    if 'MonitorG' in monitors: monitor_keys.append('MonitorG')


    print(f"--- Searching for First NaN in '{Simulated_network}' Network ---")
    
    for monitor_key in monitor_keys:
        monitor = monitors.get(monitor_key)
        
        # Determine the variable list for the current monitor
        if monitor_key == syn_monitor_key:
            vars_to_check = variable_map['SynapseMonitor']
            group_name = "Synaptic Group (S)"
        elif monitor_key == 'MonitorN':
            vars_to_check = variable_map['MonitorN']
            group_name = "Neuronal Group (N)"
        elif monitor_key == 'MonitorA':
            vars_to_check = variable_map['MonitorA']
            group_name = "Astrocytic Group (Astro)"
        elif monitor_key == 'MonitorGT':
            vars_to_check = variable_map['MonitorGT']
            group_name = "Gliotransmission Group (GT)"
        elif monitor_key == 'MonitorG':
            vars_to_check = variable_map['MonitorG']
            group_name = "Poisson Input Group (P)"
        else:
            continue # Skip if monitor not relevant or present

        if monitor is None:
            print(f"Skipping monitor: {monitor_key} (Not defined in the monitors dictionary)")
            continue
            
        print(f"\n[Checking {group_name} ({monitor_key}) ]")

        for var_name in vars_to_check:
            try:
                # 1. Access the recorded data array (units x time steps)
                data = getattr(monitor, var_name)[:]
                
                # 2. Find the time index where *any* unit has a NaN
                # np.isnan(data) -> boolean array of same shape
                # .any(axis=0) -> boolean array (True if any unit is NaN at that time step)
                nan_at_any_unit = np.isnan(data).any(axis=0)
                
                # 3. Get the indices where NaN is True
                nan_indices = np.where(nan_at_any_unit)[0]
                
                if nan_indices.size > 0:
                    first_nan_index = nan_indices[0]
                    
                    # Calculate the time of failure
                    time_of_failure = monitor.t[first_nan_index]
                    
                    print(f"  ❌ FAILURE in {var_name}:")
                    print(f"    - Index: {first_nan_index}")
                    print(f"    - Time: {time_of_failure/ms:.3f} ms")
                    nan_found = True
                    
                    # If you want to stop on the *very first* NaN in the whole run, uncomment break here
                    # return 

            except AttributeError:
                # This happens if a variable (like 'Gamma_S' in 'Neuronal' model) wasn't recorded or doesn't exist.
                print(f"  [Skipping {var_name}]: Not found or not applicable to this monitor.")
            except Exception as e:
                print(f"  [Error checking {var_name}]: {e}")


    if not nan_found:
        print("\n✅ Success: No NaN values found in any recorded variable.")
monitors_dict = {
    'MonitorS2': MonitorS2,
    'MonitorA': MonitorA,
    'MonitorN': MonitorN,
    'MonitorGT': MonitorGT,
    'SpikesN': SpikesN,
    'SpikesA': SpikesA
}


find_first_nan_index(Simulated_network, monitors_dict)

#%%


# --- Define Subplots ---
# Create a figure and a 3x1 grid of subplots (3 rows, 1 column)
fig, axes = plt.subplots(nrows=3, ncols=1, figsize=(10, 8), dpi=200, sharex=True)
fig.suptitle('Neuronal Spike Raster Plots', fontsize=16)

# --- Subplot 1: SpikesN0 ---
ax0 = axes[0]
ax0.plot(SpikesN0.t / second, SpikesN0.i, '.k', ms=2) # Increased ms for better visibility in separate plots
ax0.set_ylabel('Neuron Index', fontsize=10)
ax0.set_title('Alpha = 1', fontsize=12)
ax0.grid(True, linestyle=':', alpha=0.6)

# --- Subplot 2: SpikesN1 ---
ax1 = axes[1]
ax1.plot(SpikesN1.t / second, SpikesN1.i, '.k', ms=2)
ax1.set_ylabel('Neuron Index ', fontsize=10)
ax1.set_title('Alpha = 0.5', fontsize=12)
ax1.grid(True, linestyle=':', alpha=0.6)

# --- Subplot 3: SpikesN2 ---
ax2 = axes[2]
ax2.plot(SpikesN2.t / second, SpikesN2.i, '.k', ms=2)
ax2.set_ylabel('Neuron Index ', fontsize=10)
ax2.set_xlabel('Time (s)', fontsize=12)
ax2.set_title('Alpha = 0', fontsize=12)
ax2.grid(True, linestyle=':', alpha=0.6)

# --- Final Adjustments ---
# Improve spacing between subplots to prevent overlap
plt.tight_layout(rect=[0, 0, 1, 0.96]) # Adjust rect to make space for suptitle

plt.show()

#%%
# CHECK AR

fig, (ax1, ax2, ax3) = plt.subplots(3, 1) # Added figsize for better viewing


# ax1.plot(SpikesN.t/second,spike_trains[0],'.g', ms=5,label='Spikes')
ax1.plot(MonitorS.t / second, MonitorS[0].uar/hertz, 'r', linewidth=0.7,label=' uar')

ax1.plot(MonitorS.t / second, MonitorS1[0].uar/hertz,'--', linewidth=0.7,label='No uar')


ax1.legend()


# ax2.plot(SpikesN.t/second,spike_trains[1],'.g', ms=5,label='Spikes')
ax2.plot(MonitorS.t / second, MonitorS[100].uar/hertz, 'r', linewidth=0.7)

ax2.plot(MonitorS.t / second, MonitorS1[100].uar/hertz, '--', linewidth=0.7)



# ax3.plot(SpikesN.t/second,spike_trains[2],'.g', ms=5,label='Spikes')
ax3.plot(MonitorS.t / second, MonitorS[200].uar/hertz, 'r', linewidth=0.7)

ax3.plot(MonitorS.t / second, MonitorS1[2].uar/hertz,'--',  linewidth=0.7)

ax3.set_xlabel('Time [s]')
#%%

fig, (ax1, ax2, ax3,ax4) = plt.subplots(4, 1) # Added figsize for better viewing


# ax1.plot(SpikesN.t/second,spike_trains[0],'.g', ms=5,label='Spikes')
ax1.plot(MonitorN.t / second, MonitorN[3].V/mV, 'k', linewidth=0.7,label='Membrane Potential pre-syn')
ax1.set_xlim(0, 20)
ax1.set_ylabel('mV')
# ax1.legend()
# ax1.plot(MonitorS.t / second, MonitorS1[0].uar/hertz,'--', linewidth=0.7,label=' uar')




# ax2.plot(SpikesN.t/second,spike_trains[1],'.g', ms=5,label='Spikes')
ax2.plot(MonitorS.t / second, MonitorS[14].nar/hertz*defaultclock.dt, 'k', linewidth=0.7,label='Released vescicles')
ax2.set_xlim(0, 20)
ax2.set_ylabel('#')
# ax2.plot(MonitorS.t / second, MonitorS1[100].uar/hertz, '--', linewidth=0.7)
ax2.legend()


# ax3.plot(SpikesN.t/second,spike_trains[2],'.g', ms=5,label='Spikes')
ax3.plot(MonitorN.t / second, MonitorN[5].I_syn/nA, 'k', linewidth=0.7,label='Synaptic currents')
ax3.set_xlim(0, 20)
ax3.set_ylabel('nA')
# ax3.plot(MonitorS.t / second, MonitorS1[2].uar/hertz,'--',  linewidth=0.7)
ax3.legend()



ax4.plot(MonitorS.t / second,MonitorS[14].avail,'k',label='Available vescicles')
ax4.set_ylabel('#')
ax4.set_xlabel('Time [s]')
ax4.set_xlim(0, 20)
ax4.legend()
# fig.show()

# plt.figure(dpi=200)
# plt.plot(SpikesN.t / second, SpikesN.i, '.k', ms=0.7)
#%%

fig, (ax1, ax2, ax3,ax4) = plt.subplots(4, 1) # Added figsize for better viewing



ax1.plot(MonitorN.t / second, MonitorN[0].I_AHP / mV, 'k', linewidth=0.7)
ax1.set_ylabel('Voltage [mV]')

ax2.plot(MonitorN.t / second, MonitorN[1].I_AHP / mV, 'k', linewidth=0.7)
ax2.set_ylabel('Voltage [mV]')



ax3.plot(MonitorN.t / second, MonitorN[2].I_AHP / mV, 'k', linewidth=0.7)
ax3.set_ylabel('Voltage [mV]')



ax4.plot(MonitorN.t / second, MonitorN[3].I_AHP / mV, 'k', linewidth=0.7)
ax4.set_xlabel('Time [s]')
ax4.set_ylabel('Voltage [mV]')
fig.show()





#%%
# show()

fig, (ax1, ax2, ax3) = plt.subplots(3, 1) # Added figsize for better viewing


# ax1.plot(SpikesN.t/second,spike_trains[0],'.g', ms=5,label='Spikes')
ax1.plot(MonitorS.t / second, MonitorN[0].I_nmda/pA, 'r', linewidth=0.7,label='qar')

ax1.plot(MonitorS.t / second, MonitorN1[0].I_nmda/pA, '--', linewidth=0.7,label='No qar')


ax1.legend()


# ax2.plot(SpikesN.t/second,spike_trains[1],'.g', ms=5,label='Spikes')
ax2.plot(MonitorS.t / second, MonitorN[1].I_nmda/pA, 'r', linewidth=0.7)

ax2.plot(MonitorS.t / second, MonitorN1[1].I_nmda/pA,'--',  linewidth=0.7)



# ax3.plot(SpikesN.t/second,spike_trains[2],'.g', ms=5,label='Spikes')
ax3.plot(MonitorS.t / second, MonitorN[2].I_nmda/pA, 'r', linewidth=0.7)

ax3.plot(MonitorS.t / second, MonitorN1[2].I_nmda/pA,'--',  linewidth=0.7)

ax3.set_xlabel('Time [s]')

#%%

fig, (ax1, ax2, ax3) = plt.subplots(3, 1) # Added figsize for better viewing


# ax1.plot(SpikesN.t/second,spike_trains[0],'.g', ms=5,label='Spikes')
ax1.plot(MonitorS.t / second, MonitorN[0].I_ampa/pA, 'r', linewidth=0.7,label='qar')

ax1.plot(MonitorS.t / second, MonitorN1[0].I_ampa/pA, '--', linewidth=0.7,label='No qar')


ax1.legend()


# ax2.plot(SpikesN.t/second,spike_trains[1],'.g', ms=5,label='Spikes')
ax2.plot(MonitorS.t / second, MonitorN[1].I_ampa/pA, 'r', linewidth=0.7)

ax2.plot(MonitorS.t / second, MonitorN1[1].I_ampa/pA,'--', linewidth=0.7)



# ax3.plot(SpikesN.t/second,spike_trains[2],'.g', ms=5,label='Spikes')
ax3.plot(MonitorS.t / second, MonitorN[2].I_ampa/pA, 'r', linewidth=0.7)

ax3.plot(MonitorS.t / second, MonitorN1[2].I_ampa/pA,'--', linewidth=0.7)

ax3.set_xlabel('Time [s]')

#%% Y_S

fig, (ax1, ax2, ax3) = plt.subplots(3, 1) # Added figsize for better viewing


# ax1.plot(SpikesN.t/second,spike_trains[0],'.g', ms=5,label='Spikes')
ax1.plot(MonitorS.t / second, MonitorS[0].Y_S/mmole, 'r', linewidth=0.7,label='qar')
ax1.plot(MonitorS.t / second, MonitorS1[0].Y_S/mmole, 'b','--', linewidth=0.7,label='NO qar')
ax1.legend()



# ax2.plot(SpikesN.t/second,spike_trains[1],'.g', ms=5,label='Spikes')
ax2.plot(MonitorS.t / second, MonitorS[30].Y_S/mmole,'r', linewidth=0.7)
ax2.plot(MonitorS.t / second, MonitorS1[30].Y_S/mmole,'b','--', linewidth=0.7)





# ax3.plot(SpikesN.t/second,spike_trains[2],'.g', ms=5,label='Spikes')
ax3.plot(MonitorS.t / second, MonitorS[100].Y_S/mmole,'r', linewidth=0.7)
ax3.plot(MonitorS.t / second, MonitorS1[100].Y_S/mmole,'b','--', linewidth=0.7)


ax3.set_xlabel('Time [s]')




#%%
# CHECK AR
fig, (ax1, ax2, ax3) = plt.subplots(3, 1) # Added figsize for better viewing


# ax1.plot(SpikesN.t/second,spike_trains[0],'.g', ms=5,label='Spikes')
ax1.plot(MonitorS.t / second, MonitorN[0].I_ampa/pA, 'r', linewidth=0.7,label='I_ampa')

ax1.plot(MonitorS.t / second, MonitorN[0].I_nmda/pA, '--', linewidth=0.7,label='I_nmda')


ax1.legend()


# ax2.plot(SpikesN.t/second,spike_trains[1],'.g', ms=5,label='Spikes')
ax2.plot(MonitorS.t / second, MonitorN[1].I_ampa/pA, 'r', linewidth=0.7)

ax2.plot(MonitorS.t / second, MonitorN[1].I_nmda/pA,'--', linewidth=0.7)



# ax3.plot(SpikesN.t/second,spike_trains[2],'.g', ms=5,label='Spikes')
ax3.plot(MonitorS.t / second, MonitorN[2].I_ampa/pA, 'r', linewidth=0.7)

ax3.plot(MonitorS.t / second, MonitorN[2].I_nmda/pA,'--', linewidth=0.7)

ax3.set_xlabel('Time [s]')


#%%
fig, (ax1, ax2, ax3) = plt.subplots(3, 1) # Added figsize for better viewing


# ax1.plot(SpikesN.t/second,spike_trains[0],'.g', ms=5,label='Spikes')
ax1.plot(MonitorS.t / second, MonitorS[0].r_ampa/hertz, 'r', linewidth=0.7,label='r_ampa')

ax1.plot(MonitorS.t / second, MonitorS[0].r_nmda/hertz,'--', linewidth=0.7,label='r_nmda')


ax1.legend()


# ax2.plot(SpikesN.t/second,spike_trains[1],'.g', ms=5,label='Spikes')
ax2.plot(MonitorS.t / second, MonitorS[30].r_ampa/hertz, 'r', linewidth=0.7)

ax2.plot(MonitorS.t / second, MonitorS[30].r_nmda/hertz, '--', linewidth=0.7)



# ax3.plot(SpikesN.t/second,spike_trains[2],'.g', ms=5,label='Spikes')
ax3.plot(MonitorS.t / second, MonitorS[100].r_ampa/hertz, 'r', linewidth=0.7)

ax3.plot(MonitorS.t / second, MonitorS[100].r_nmda/hertz,'--',  linewidth=0.7)

ax3.set_xlabel('Time [s]')


fig, (ax1, ax2, ax3) = plt.subplots(3, 1) # Added figsize for better viewing


# ax1.plot(SpikesN.t/second,spike_trains[0],'.g', ms=5,label='Spikes')
ax1.plot(MonitorS.t / second, MonitorS[0].r_Sr/hertz, 'r', linewidth=0.7,label='r_Sr')

ax1.plot(MonitorS.t / second, MonitorS[0].r_Ar/hertz,'--', linewidth=0.7,label='r_Ar')


ax1.legend()


# ax2.plot(SpikesN.t/second,spike_trains[1],'.g', ms=5,label='Spikes')
ax2.plot(MonitorS.t / second, MonitorS[30].r_Sr/hertz, 'r', linewidth=0.7)

ax2.plot(MonitorS.t / second, MonitorS[30].r_Ar/hertz, '--', linewidth=0.7)



# ax3.plot(SpikesN.t/second,spike_trains[2],'.g', ms=5,label='Spikes')
ax3.plot(MonitorS.t / second, MonitorS[100].r_Sr/hertz, 'r', linewidth=0.7)

ax3.plot(MonitorS.t / second, MonitorS[100].r_Ar/hertz,'--',  linewidth=0.7)

ax3.set_xlabel('Time [s]')












#%%
# ------- SYNAPSES -------

fig, (ax1, ax2, ax3) = plt.subplots(3, 1) # Added figsize for better viewing


# ax1.plot(SpikesN.t/second,spike_trains[0],'.g', ms=5,label='Spikes')
ax1.plot(MonitorS.t / second, MonitorS[0].usr, 'r', linewidth=0.7,label='Ready-to-relase resources')

ax1.plot(MonitorS.t / second, MonitorS[0].x_S, 'c', linewidth=0.7,label='Available resources')


ax1.legend()


# ax2.plot(SpikesN.t/second,spike_trains[1],'.g', ms=5,label='Spikes')
ax2.plot(MonitorS.t / second, MonitorS[30].usr, 'r', linewidth=0.7)

ax2.plot(MonitorS.t / second, MonitorS[30].x_S, 'c', linewidth=0.7)



# ax3.plot(SpikesN.t/second,spike_trains[2],'.g', ms=5,label='Spikes')
ax3.plot(MonitorS.t / second, MonitorS[100].usr, 'r', linewidth=0.7)

ax3.plot(MonitorS.t / second, MonitorS[100].x_S, 'c', linewidth=0.7)

ax3.set_xlabel('Time [s]')
# fig.show()

# plt.figure(dpi=200)
# plt.plot(SpikesN.t / second, SpikesN.i, '.k', ms=0.7)

# show()

#%%

# Synaptically released glutamate 

fig, (ax1, ax2, ax3) = plt.subplots(3, 1) # Added figsize for better viewing




ax1.plot(MonitorS.t / second, MonitorS[0].Y_S, 'k', linewidth=0.7,label='[Glu]')
# ax1.plot(SpikesN.t/second,SpikesN[0].i+3,'.g', ms=5,label='Spikes')
ax1.set_ylabel('Mol')
ax1.legend()


ax2.plot(MonitorS.t / second, MonitorS[1].Y_S, 'k', linewidth=0.7)
# ax2.plot(SpikesN.t/second,SpikesN[1].i+3,'.g', ms=5)
ax2.set_ylabel('Mol')


ax3.plot(MonitorS.t / second, MonitorS[2].Y_S, 'k', linewidth=0.7)
# ax3.plot(SpikesN.t/second,SpikesN[2].i+3,'.g', ms=5)
# fig.show()
ax3.set_ylabel('Mol')
ax3.set_xlabel('Time [s]')
# plt.figure(dpi=200)
# plt.plot(SpikesN.t / second, SpikesN.i, '.k', ms=0.7)

show()

#%%

# ASTROCYTIC released glutamate 

fig, (ax1, ax2, ax3) = plt.subplots(3, 1) # Added figsize for better viewing




ax1.plot(MonitorGT.t / second, MonitorGT[0].G_A, 'k', linewidth=0.7,label='[Glu]')
# ax1.plot(SpikesN.t/second,SpikesN[0].i+3,'.g', ms=5,label='Spikes')
ax1.set_ylabel('Mol')
ax1.legend()


ax2.plot(MonitorGT.t / second, MonitorGT[1].G_A, 'k', linewidth=0.7)
# ax2.plot(SpikesN.t/second,SpikesN[1].i+3,'.g', ms=5)
ax2.set_ylabel('Mol')


ax3.plot(MonitorGT.t / second, MonitorGT[2].G_A, 'k', linewidth=0.7)
# ax3.plot(SpikesN.t/second,SpikesN[2].i+3,'.g', ms=5)
# fig.show()
ax3.set_ylabel('Mol')
ax3.set_xlabel('Time [s]')
# plt.figure(dpi=200)
# plt.plot(SpikesN.t / second, SpikesN.i, '.k', ms=0.7)

show()

#%%

# SYNAPTIC CURRENTS



fig, (ax1, ax2, ax3) = plt.subplots(3, 1) # Added figsize for better viewing



ax1.plot(MonitorN.t/second, MonitorN[1].I_syn, 'k', label='I_syn')
ax1.plot(MonitorN.t/second, MonitorN[1].I_ampa, 'r', label='I_ampa')
ax1.plot(MonitorN.t/second, MonitorN[1].I_nmda, 'b', label='I_nmda')
# ax1.plot(SpikesN.t/second, SpikesN[0].i+1e-11, '.g', ms=5, label='Spikes') # Or a more descriptive label like 'Neuron SpikesN'
ax1.set_ylabel('A')
ax1.legend()


ax2.plot(MonitorN.t/second, MonitorN[2].I_syn, 'k')
ax2.plot(MonitorN.t/second, MonitorN[2].I_ampa, 'r')
ax2.plot(MonitorN.t/second, MonitorN[2].I_nmda, 'b')
# ax2.plot(SpikesN.t/second,SpikesN[1].i+1e-11,'.g', ms=5)
ax2.set_ylabel('A')


ax3.plot(MonitorN.t/second, MonitorN[3].I_syn, 'k')
ax3.plot(MonitorN.t/second, MonitorN[3].I_ampa, 'r')
ax3.plot(MonitorN.t/second, MonitorN[3].I_nmda, 'b')
# ax3.plot(SpikesN.t/second,SpikesN[2].i+1e-11,'.g', ms=5)
# fig.show()
ax3.set_ylabel('A')
ax3.set_xlabel('Time [s]')
# plt.figure(dpi=200)
# plt.plot(SpikesN.t / second, SpikesN.i, '.k', ms=0.7)

show()

#%%

# ASTROCYTIC NETWORK

color1 = [0.4196078431372549, 0.47843137254901963, 0.5607843137254902] 
color2 =[0.9686274509803922, 0.9333333333333333, 0.4980392156862745]
color3 = [0.7411764705882353, 0.30980392156862746, 0.4235294117647059]

fig, (ax1, ax2, ax3) = plt.subplots(3, 1) # Added figsize for better viewing

maxC = np.max(MonitorA[0].C)
maxI = np.max(MonitorA[0].I)
ax1.plot(MonitorA.t/second, MonitorA[0].C/umole, 'k', label='Calcium',color=color1)
# ax1.plot(MonitorA.t/second, MonitorA[0].I/maxI, 'r', label='IP_3',color=color2)
ax1.plot(MonitorA.t/second, MonitorA[0].Gamma_A, 'b', label='Fraq. bounded receptors',color=color3)
# ax1.plot(SpikesN.t/second, SpikesN[0].i+1e-11, '.g', ms=5, label='Spikes') # Or a more descriptive label like 'Neuron SpikesN'
ax1.set_ylabel('A')
ax1.legend()


ax2.plot(MonitorA.t/second, MonitorA[1].C/umole, 'k',color=color1)
# ax2.plot(MonitorA.t/second, MonitorA[1].I/maxI, 'r',color=color2)
ax2.plot(MonitorA.t/second, MonitorA[1].Gamma_A, 'b',color=color3)
# ax2.plot(SpikesN.t/second,SpikesN[1].i+1e-11,'.g', ms=5)
ax2.set_ylabel('A')


ax3.plot(MonitorA.t/second, MonitorA[2].C/umole, 'k',color=color1)
# ax3.plot(MonitorA.t/second, MonitorA[2].I/maxI, 'r',color=color2)
ax3.plot(MonitorA.t/second, MonitorA[2].Gamma_A, 'b',color=color3)
# ax3.plot(SpikesN.t/second,SpikesN[2].i+1e-11,'.g', ms=5)
# fig.show()
ax3.set_ylabel('')
ax3.set_xlabel('Time [s]')
# plt.figure(dpi=200)
# plt.plot(SpikesN.t / second, SpikesN.i, '.k', ms=0.7)

show()


#%%
# --------- ASTROCYTE RASTER ---------

%matplotlib
plt.figure()
plt.plot(SpikesA2.t / second, SpikesA2.i, '.k', ms=4)
plt.title("Astro Raster Plot")
plt.xlabel("Time [s]")
plt.ylabel("Astrocyte")
plt.show()

#%%
# --------- ASTROCYTE CALCIUM ---------
# Get stimulated astrocytes idx
# Stim_astro = list(Glu_Input.j)

# plt.figure()

# for astro in range(Astro.N):
#     if astro in Stim_astro:
      
#         plt.plot(MonitorA.t/second,MonitorA[astro].C/mmole + astro*0.0001,c='r')

        
#     else:    
        
#         plt.plot(MonitorA.t/second,MonitorA[astro].C/mmole + astro*0.0001,c='k')
    
# plt.show()    
    

# Generate the array of traces
Trace_astro = []

for astro in range(Astro.N):
    
    Trace_astro.append(list(MonitorA[astro].C/mmole))
    
Trace_astro = np.vstack(Trace_astro)    

#
min_trace = np.min(Trace_astro)
max_trace = np.max(Trace_astro)

Trace_astro_norm= normalize_to_range(Trace_astro, min_trace, max_trace, 0, 1)
#
from matplotlib.collections import LineCollection
import resampy
import matplotlib.colors as mcolors
downs_factor = 100
# Resample (for comp. feasability)
Trace_astro_norm_res = resampy.resample(Trace_astro_norm, len(Trace_astro_norm[0,:]), len(Trace_astro_norm[0,:])/downs_factor,axis=1)


# Create color map
colors = [(0, 0, 0), (1, 0, 0)] # This defines the start and end colors
%matplotlib
t_vec = np.linspace(0,len(Trace_astro_norm_res[0,:]),len(Trace_astro_norm_res[0,:]))
# Create the colormap from this list of colors
my_cmap = mcolors.LinearSegmentedColormap.from_list("BlackRed", colors)
plt.figure()
for astro in range(Astro.N):
    plt.scatter(t_vec, np.ones(len(t_vec))*astro, c=Trace_astro_norm_res[astro,:], cmap='inferno', s=7,
                vmin=np.min(Trace_astro_norm_res),  # Set the global minimum for the color scale
                vmax=np.max(Trace_astro_norm_res))
plt.title("Calcium traces")
plt.xlabel("Time [ms]")
plt.yticks([])
plt.colorbar(label="Normalized [C]")
plt.show()


#%%

# --------------- ELECTRODE RECORDINGS NEURONAL CULTURE ---------------
Traces,MEA_dict = Electrode_traces(pitch,pitch_recsites,shift,N,MonitorN,electrode_dist,neuron_radius,electrode_radius)

clock_dt = defaultclock.dt 

Raster,Raster_array = get_Raster(Traces,clock_dt)

# ------- SAVE -------
#%%
# # Meta-data dict
Meta_data_nn ={
    
    'Simulation_time': simtime/second, #[sec]
    
    'fs': 1/(clock_dt/second), # [Hz]
    
    'Raster_array': Raster_array,
    
    'Electrode_traces': Traces,
    
    'MEA_dict': MEA_dict,
    
    

    
    }


os.chdir(Out_path)

np.save('Simulation_dict.npy', Meta_data_nn, allow_pickle=True)
np.save('Electrode_traces.npy', Traces, allow_pickle=True)
np.save('MEA_dict.npy', MEA_dict, allow_pickle=True)


#%%
# ----- Neuronal Dynamics ------
Type_Neruonal_dynamics = 'Cumulative'


if Type_Neruonal_dynamics == 'PCA':
    Projected_trajectories,Variance_explained,fs_downsampled = Neuronal_traces_simulation(Raster_array,Type = Type_Neruonal_dynamics,t_rec = simtime/second, fs = 1/(clock_dt/second), w_size = 0.12, overlap = 0.06, 
                                                                                            bin_size_s = 0.05, Isolate_NB = False, Gaussian_window = 0.002,
                                                                                             Visible = True)

elif Type_Neruonal_dynamics == 'Cumulative':

    Cumulative,fs_downsampled = Neuronal_traces_simulation(Raster_array,Type = Type_Neruonal_dynamics,t_rec = simtime/second, fs = 1/(clock_dt/second), w_size = 0.01, overlap = 0.06, 
                                                                                            bin_size_s = 0.05, Isolate_NB = False, Gaussian_window = 0.021,
                                                                                             Visible = True)


#%%

# --------------- ASTROCYTIC NETWORK ---------------

Grid = Get_12grid(pitch)

MEA_dict = Recording_sites(pitch_recsites,shift)

# --- Plot Device + Neurons
%matplotlib

Plot_CultureDevice(Grid,A,Na)

#%%

# ----------- FULL -------------
# Comparison between coupled and decopuled Neuron-Astro networks i synaptic release Y_S

fig, (ax1, ax2, ax3) = plt.subplots(3, 1) # Added figsize for better viewing




ax1.plot(MonitorS1.t / second, MonitorS1[0].Y_S/umole, 'k', linewidth=3,label='Coupled')
ax1.plot(MonitorS2.t / second, MonitorS2[0].Y_S/umole, '--','c', linewidth=2,label='Decoupled')
# ax1.plot(SpikesN.t/second,SpikesN[0].i+3,'.g', ms=5,label='Spikes')
ax1.set_ylabel('Mol')
ax1.legend()


ax2.plot(MonitorS1.t / second, MonitorS1[1].Y_S/umole, 'k', linewidth=3)
ax2.plot(MonitorS2.t / second, MonitorS2[1].Y_S/umole, '--','c', 'k', linewidth=2)
# ax2.plot(SpikesN.t/second,SpikesN[1].i+3,'.g', ms=5)
ax2.set_ylabel('Mol')


ax3.plot(MonitorS1.t / second, MonitorS1[2].Y_S/umole, 'k', linewidth=3)
ax3.plot(MonitorS2.t / second, MonitorS2[2].Y_S/umole, '--','c', 'k', linewidth=2)
# ax3.plot(SpikesN.t/second,SpikesN[2].i+3,'.g', ms=5)
# fig.show()
ax3.set_ylabel('Mol')
ax3.set_xlabel('Time [s]')
# plt.figure(dpi=200)
# plt.plot(SpikesN.t / second, SpikesN.i, '.k', ms=0.7)

show()

