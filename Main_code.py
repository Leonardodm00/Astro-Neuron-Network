
# -*- coding: utf-8 -*-
"""
Created on Tue Jul 22 17:06:17 2025

@author: leona
"""


import matplotlib.pyplot as plt
from brian2 import *
from scipy.spatial import KDTree

import os


os.chdir(r'C:\Users\Admin\Desktop\CURRENTS')
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
Adaptation = False


# ------- Astrocytes -------
oscillations = 'AM'










# ------------------------- PARAMETERS -------------------------

# --------- SIMULATION -----------
simtime = 40 * second               # simulation time
# transient = 3 * second              # time omitted as transient
sed = 39                             # random number seed
devices.device.seed(sed)            # set the seed for all the random number realisations


Simulated_network = 'Neuronal' # Astrocytic/Neuronal/Full


# --------- NEURON -----------
Nn = 250


# --------- SYNAPTIC -----------

# ----- Connectivity -----
#TODO: for now is not needed connections will be
#   randomly set
ADJ_neuro = np.array(([0,1,0,0],
                      [0,0,1,0],
                      [0,0,0,1],
                      [0,0,0,0]))


# --------- ASTROCYTE and GJ ----------- 
'''
Given the nature of the link hte adjency matrix is ALWAYS symmetric

'''
Na = 3
ADJ_astro = np.array(([0,1,0],
                      [1,0,1],
                      [0,1,0]))



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
radius = 15 #[um]

pitch_recsites = 7.5 # [um]  
shift = 11.25 # [um]

electrode_dist = 300 # [um]


# ------------------------- GROUPS BUILD-UP -------------------------

if Simulated_network == 'Full':
    # --------- NEURON and SYNAPSE -----------
    N,S = Neuronal_Network(Nn,ADJ_neuro, RandomKinetics=RandomKinetics, OnlyExc=OnlyExc,
                           Syn_Currents_model=Syn_Currents_model,add_delay=add_delay,
                           delay_mode=delay_mode,Max_delay=Max_delay,ics=ics,
                           std_pers=std_pers, Simulated_network=Simulated_network,Decay_type=Decay_type,
                           synapse_type=synapse_type)
    
    
    # ----- SET POSITIONS AND CONNECTIONS -----
    # Position neurons on a grid
    grid_dist = 45 * umeter
    N.x = '(i % Nn) * grid_dist'
    N.y = '(i // Nn) * grid_dist'
    
    
    
    
    
    
    # --------- ASTROCYTE -----------
    A,GJ = Astrocyte_Group(N_astro,ADJ_astro)
    
    
    # --------- GLIOTRANSMISSION ----------- 
    GT = Gliotransmission(N_astro,ics,Astro)
    
    
    # --------- ASTRO-NEURON LINKS ----------- 
    # --- Synapse to astro ---
    StoA = Synapse_to_astro(S,A,ADJ_SynAstro)
    
    # --- Astro to synapse ---
    AtoS = Astro_to_Syn(GT,S,ADJ_AstroSyn)
    
    
    
    
elif Simulated_network == 'Neuronal':

    # --------- NEURON and SYNAPSE -----------
    N,S = Neuronal_Network(Nn,ADJ_neuro, RandomKinetics=RandomKinetics, OnlyExc=OnlyExc,
                           Syn_Currents_model=Syn_Currents_model,add_delay=add_delay,
                           delay_mode=delay_mode,Max_delay=Max_delay,ics=ics,
                           std_pers=std_pers, Simulated_network=Simulated_network,Decay_type=Decay_type,synapse_type = synapse_type)
    
    
    # ----- SET POSITIONS AND CONNECTIONS -----
    # Position neurons on a grid
    grid_dist = 45 * umeter
    N.x = '(i % Nn) * grid_dist'
    N.y = '(i // Nn) * grid_dist'
    

    
    
    
elif Simulated_network == 'Astrocytic':   
    
    # --------- ASTROCYTE -----------
    Astro, GJ,P,Glu_Input = Astrocyte_Group(Na,ADJ_astro,Simulated_network)
    
    
    
    
    
    






# ------------------------- NETWORK SIMULATION -------------------------

# --- Monitors ---
recording_stringN = ['V','I_syn','I_ampa','I_nmda']
recording_stringS = ['u_S','x_S','Y_S']
recording_stringA = ['C','I','Gamma_A','I_coupling_tot']
recording_stringGT = ['G_A','x_A']


if Simulated_network == 'Full':
    
    MonitorS = StateMonitor(S, recording_stringS, record=True)
    MonitorA = StateMonitor(Astro, recording_stringA, record=True)
    MonitorN = StateMonitor(N, recording_stringN, record=True)
    SpikesN = SpikeMonitor(N)
    SpikesA = SpikeMonitor(A)
    

elif Simulated_network == 'Neuronal':
    # MonitorS = StateMonitor(S, recording_stringS, record=True)
    MonitorN = StateMonitor(N, recording_stringN, record=True)
    SpikesN = SpikeMonitor(N)
    
    
elif Simulated_network == 'Astrocytic': 
    MonitorG = StateMonitor(Glu_Input, ['Y_bias_in'], record=True)
    MonitorA = StateMonitor(Astro, recording_stringA, record=True)
    SpikesP = SpikeMonitor(Astro)
    SpikesP = SpikeMonitor(P)


# --- Collect and add monitors ---


net_ = Network(collect())  # automatically include all the stated groups
net_.run(simtime,report='text', profile=True)



#%%
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
ax1.plot(MonitorA.t/second, MonitorA[0].C/maxC, 'k', label='Calcium',color=color1)
ax1.plot(MonitorA.t/second, MonitorA[0].I/maxI, 'r', label='IP_3',color=color2)
ax1.plot(MonitorA.t/second, MonitorA[0].Gamma_A, 'b', label='Fraq. bounded receptors',color=color3)
# ax1.plot(SpikesN.t/second, SpikesN[0].i+1e-11, '.g', ms=5, label='Spikes') # Or a more descriptive label like 'Neuron SpikesN'
ax1.set_ylabel('A')
ax1.legend()


ax2.plot(MonitorA.t/second, MonitorA[1].C/maxC, 'k',color=color1)
ax2.plot(MonitorA.t/second, MonitorA[1].I/maxI, 'r',color=color2)
ax2.plot(MonitorA.t/second, MonitorA[1].Gamma_A, 'b',color=color3)
# ax2.plot(SpikesN.t/second,SpikesN[1].i+1e-11,'.g', ms=5)
ax2.set_ylabel('A')


ax3.plot(MonitorA.t/second, MonitorA[2].C/maxC, 'k',color=color1)
ax3.plot(MonitorA.t/second, MonitorA[2].I/maxI, 'r',color=color2)
ax3.plot(MonitorA.t/second, MonitorA[2].Gamma_A, 'b',color=color3)
# ax3.plot(SpikesN.t/second,SpikesN[2].i+1e-11,'.g', ms=5)
# fig.show()
ax3.set_ylabel('')
ax3.set_xlabel('Time [s]')
# plt.figure(dpi=200)
# plt.plot(SpikesN.t / second, SpikesN.i, '.k', ms=0.7)

show()


#%%

fig.show()

plt.figure(dpi=200)
plt.plot(SpikesA.t / second, SpikesA.i, '.k', ms=0.7)

show()

#%%

# --------------- ELECTRODE RECORDINGS ---------------

Grid = Get_12grid(pitch)

MEA_dict = Recording_sites(pitch_recsites,shift)

Traces = Electrode_recording(MEA_dict,N,MonitorN,electrode_dist)










