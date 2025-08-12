# -*- coding: utf-8 -*-
"""
Created on Tue Aug 12 17:53:15 2025

@author: Admin
"""


import matplotlib.pyplot as plt
from brian2 import *


import os


os.chdir(r'C:\Users\Admin\Desktop\Leonardo\ASN')
import ASN_fun


'''


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





REFERENCES:
    1) Distance dependent prob of conneciton = Estimating neuronal connectivity from axonal and dendritic density fields
    2) Astrocyte connections: A Computational Model of Interactions Between Neuronal and Astrocytic Networks: The Role of Astrocytes in the Stability of the Neuronal Firing Rate

    
    
    
TODO:
    1) Optimize and parallelize Electrode_recording
'''












start_scope()
# ------------------------- SET OPTIONS -------------------------

# ------- Synapses -------
synapse_type='neutral'
ics=None 
dt=None            

postc_sic='double-exp'
Decay_type = 'Double_exp'
sic=None 
delay=None
RandomKinetics = False 
OnlyExc = True
std_pers =0.01

Syn_Currents_model = 'TM-coupled' # 'Kinetic','Nina','TM-coupled'

Max_delay = 25 *ms
add_delay = False
delay_mode = 'random'



# ------- Neurons -------
Adaptation = True


# ------- Astrocytes -------
oscillations = 'AM'










# ------------------------- PARAMETERS -------------------------

# --------- SIMULATION -----------
simtime = 20 * second               # simulation time
# transient = 3 * second              # time omitted as transient
sed = 39                             # random number seed
devices.device.seed(sed)            # set the seed for all the random number realisations


Simulated_network = 'Neuronal' # Astrocytic/Neuronal/Full


# --------- NEURON -----------
Nn = 100
neuron_radius = 9 #[um]
# --------- SYNAPTIC -----------

# ----- Connectivity -----
# Can be directly an ADJ or a string: Random, Distance
# Connection_neuro = np.array(([0,1,0,0],
#                               [0,0,1,0],
#                               [0,0,0,1],
#                               [0,0,0,0]))
Connection_neuro = 'Distance'

# --------- ASTROCYTE and GJ ----------- 
'''
Given the nature of the link hte adjency matrix is ALWAYS symmetric

'''
Na = 3

# ----- Connectivity -----
# Can be directly an ADJ or a string: Distance
# Connection_astro= np.array(([0,1,0],
#                             [1,0,1],
#                             [0,1,0]))
Connection_astro = 'Distance'


# --------- GLIOTRANSMISSION ----------- 


# --------- ASTRO-NEURON LINKS ----------- 
# --- Synapse to astro ---
ADJ_SynAstro = np.array(([1,0,0],
                         [0,1,0],
                         [0,0,1]))


# --- Astro to synapse ---
ADJ_AstroSyn = np.array(([1,0,0],
                         [0,1,0],
                         [0,0,1]))


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
    N,S = Neuronal_Network(Nn,Connection_neuro, RandomKinetics=RandomKinetics, OnlyExc=OnlyExc,
                           Syn_Currents_model=Syn_Currents_model,add_delay=add_delay,
                           delay_mode=delay_mode,Max_delay=Max_delay,ics=ics,
                           std_pers=std_pers, Simulated_network=Simulated_network,Decay_type=Decay_type,
                           synapse_type=synapse_type)
    
    

    
    
    
    
    
    
    # --------- ASTROCYTE -----------
    Astro,GJ = Astrocyte_Group(Na,Connection_astro,Simulated_network,sed)
    
    
    
    
    # --------- GLIOTRANSMISSION ----------- 
    GT = Gliotransmission(Na,ics,Astro)
    
    
    # --------- ASTRO-NEURON LINKS ----------- 
    # --- Synapse to astro ---
    StoA = Synapse_to_astro(S,Astro,ADJ_SynAstro)
    
    # --- Astro to synapse ---
    AtoS = Astro_to_Syn(GT,S,ADJ_AstroSyn)
    
    
    
    
elif Simulated_network == 'Neuronal':

    # --------- NEURON and SYNAPSE -----------
    N,S = Neuronal_Network(Nn,Connection_neuro, RandomKinetics=RandomKinetics, OnlyExc=OnlyExc,
                           Syn_Currents_model=Syn_Currents_model,add_delay=add_delay,
                           delay_mode=delay_mode,Max_delay=Max_delay,ics=ics,
                           std_pers=std_pers, Simulated_network=Simulated_network,Decay_type=Decay_type,synapse_type = synapse_type)
    
    
   
    

    
    
    
elif Simulated_network == 'Astrocytic':   
 
    '''
    Glutamate stimulation is delivered to randomly choosen astrocytes in number 
    N_stim in the parameters.
    '''
    # --------- ASTROCYTE -----------
    Astro, GJ,P,Glu_Input = Astrocyte_Group(Na,Connection_astro,Simulated_network,sed)
    



# ------------------------- NETWORK SIMULATION -------------------------

# --- Monitors ---
# recording_stringN = ['V','I_syn','I_ampa','I_nmda','I_cell']
recording_stringN = ['V','I_cell']
recording_stringS = ['u_S','x_S','Y_S']
recording_stringA = ['C','I','Gamma_A','I_coupling_tot','Y_extra']
recording_stringGT = ['G_A','x_A']


if Simulated_network == 'Full':
    
    MonitorS2 = StateMonitor(S, recording_stringS, record=True)
    MonitorA = StateMonitor(Astro, recording_stringA, record=True)
    MonitorN = StateMonitor(N, recording_stringN, record=True)
    MonitorGT = StateMonitor(GT, recording_stringGT, record=True)
    SpikesN = SpikeMonitor(N)
    SpikesA = SpikeMonitor(Astro)
    

elif Simulated_network == 'Neuronal':
    # MonitorS = StateMonitor(S, recording_stringS, record=True)
    MonitorN = StateMonitor(N, recording_stringN, record=True)
    SpikesN = SpikeMonitor(N)
    
    
elif Simulated_network == 'Astrocytic': 
    MonitorG = StateMonitor(Glu_Input, ['Y_bias_in'], record=True)
    MonitorA = StateMonitor(Astro, recording_stringA, record=True)
    SpikesA = SpikeMonitor(Astro)
    SpikesP = SpikeMonitor(P)


# --- Collect and add monitors ---


net_ = Network(collect())  # automatically include all the stated groups
net_.run(simtime,report='text', profile=True)




spike_trains = SpikesN.spike_trains()
# --------------------- PLOTS ---------------------
# ------- NEURONS -------
%matplotlib
fig, (ax1, ax2, ax3,ax4) = plt.subplots(4, 1) # Added figsize for better viewing



ax1.plot(MonitorN.t / second, MonitorN[0].V / mV, 'k', linewidth=0.7)
ax1.set_ylabel('Voltage [mV]')

ax2.plot(MonitorN.t / second, MonitorN[1].V / mV, 'k', linewidth=0.7)
ax2.set_ylabel('Voltage [mV]')



ax3.plot(MonitorN.t / second, MonitorN[2].V / mV, 'k', linewidth=0.7)
ax3.set_ylabel('Voltage [mV]')



ax4.plot(MonitorN.t / second, MonitorN[3].V / mV, 'k', linewidth=0.7)
ax4.set_xlabel('Time [s]')
ax4.set_ylabel('Voltage [mV]')
fig.show()

plt.figure(dpi=200)
plt.plot(SpikesN.t / second, SpikesN.i, '.k', ms=0.7)

show()

#%%

# ------- SYNAPSES -------

fig, (ax1, ax2, ax3) = plt.subplots(3, 1) # Added figsize for better viewing


# ax1.plot(SpikesN.t/second,spike_trains[0],'.g', ms=5,label='Spikes')
ax1.plot(MonitorS.t / second, MonitorS[0].u_S, 'r', linewidth=0.7,label='Ready-to-relase resources')

ax1.plot(MonitorS.t / second, MonitorS[0].x_S, 'c', linewidth=0.7,label='Available resources')


ax1.legend()


# ax2.plot(SpikesN.t/second,spike_trains[1],'.g', ms=5,label='Spikes')
ax2.plot(MonitorS.t / second, MonitorS[1].u_S, 'r', linewidth=0.7)

ax2.plot(MonitorS.t / second, MonitorS[1].x_S, 'c', linewidth=0.7)



# ax3.plot(SpikesN.t/second,spike_trains[2],'.g', ms=5,label='Spikes')
ax3.plot(MonitorS.t / second, MonitorS[2].u_S, 'r', linewidth=0.7)

ax3.plot(MonitorS.t / second, MonitorS[2].x_S, 'c', linewidth=0.7)

ax3.set_xlabel('Time [s]')
fig.show()

plt.figure(dpi=200)
plt.plot(SpikesN.t / second, SpikesN.i, '.k', ms=0.7)

show()

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
plt.plot(SpikesA.t / second, SpikesA.i, '.k', ms=4)
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
#%%
clock_dt = defaultclock.dt 

def get_Raster(Traces,dt,low_f=100,Visible=True):
    from scipy import signal

    from scipy.signal import find_peaks
    '''
    Alternatively an elliptic filter can be used.
    Elliptic filters offer the steepest possible rolloff between the passband and stopband for a given filter order.
    This makes them highly efficient for applications that require a sharp frequency cutoff. 
    However, this superior performance comes at the cost of ripples in both the passband and the stopband.
    
    
    '''
    
        # set up a filter to filter the voltage signal
    fs = 1 / (dt / second)
    fc = low_f                                          # Cut-off frequency of the filter
    w = fc / (fs / 2)                                   # Normalize the frequency
    b, a = signal.butter(2, w, 'high')
    
    APs = []
    # voltagetraces = zeros((len(Traces),len(Traces[0])))
    Raster = zeros((len(Traces),len(Traces[0])))
    k = 0 
    for Trace_temp in Traces:
        
        # Subtract the mean
        Trace_temp = Trace_temp - np.mean(Trace_temp)
        Voltagefilt = signal.filtfilt(b, a, Trace_temp)  # high pass filter
        threshold = 4 * np.std(Voltagefilt)      #threshold to detect APs
        APstemp, _ = find_peaks(abs(Voltagefilt), height=threshold)
        for j in range(len(APstemp)):
            APs = np.append(APs, [k,APstemp[j]])
        # voltagetraces[k, :] = Voltagefilt

        k = k+1
        
        Raster[k,APstemp] = 1
        
        
        
        
    if Visible == 'True':
        
        
            # Create the plot
        plt.figure()
        
        # Plot the unit indices (y-axis) against the spike times (x-axis)
        plt.scatter(APs[:, 1]/sec, APs[:, 0], s=5, marker='|')
        
        # Customize the plot
        plt.title('Spiking Activity (Raster Plot)')
        plt.xlabel('Time (s)')
        plt.ylabel('Channel')
        plt.yticks(np.unique(spike_data[:, 0]))  # Set y-ticks to be the unique unit indices
        plt.grid(True)
        plt.show()
        
    
    
        
        
    
    return Raster



Raster = get_Raster(Traces,clock_dt)



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













