# -*- coding: utf-8 -*-
"""
Created on Sat Sep  6 09:16:32 2025

@author: Admin
"""

'''
Phenomenological model to study coupled astrocytic and neuronal dynamics.

Neurons: Izhikevich model

Synapses: AMPA and GABA are modeled as a single decay exponential while the NMDA with 
        a double decay exponential. Params: Peak amplitude, tau decay and tau rise
        (for NMDA). The synaptic amplitude (delta) is scaled by x_d which accounts for 
        depression mechanisms

Astrocytes: ...




NOTES:
    1) The inhibitory neruons haven't been implemented yet.

'''


import matplotlib.pyplot as plt
from brian2 import *

from brian2 import clear_cache


import os


os.chdir(r'C:\Users\Admin\Desktop\Leonardo\ASN\ASN_pheno')
import ASN_fun_pheno

# Saving 
Out_path = r'C:\Users\Admin\Desktop\Leonardo\ASN\Output Temp'





# Clear the cache for the 'cython' code generation target
# clear_cache('cython') 
start_scope()
# ------------------------- SET OPTIONS -------------------------



# ------------------------- PARAMETERS -------------------------

# --------- SIMULATION -----------
simtime = 10 * second               # simulation time
# transient = 3 * second              # time omitted as transient
sed = 39                             # random number seed
devices.device.seed(sed)            # set the seed for all the random number realisations


Simulated_network = 'Neuronal' # Astrocytic/Neuronal/Full


# --------- NEURON -----------
Nn = 10
neuron_radius = 9 #[um]
I_inj = 6
sigma_noise = 2
# --------- SYNAPTIC -----------

# ----- Connectivity -----
# Can be directly an ADJ or a string: Random, Distance
# Connection_neuro = np.array(([0,1,0,0],
#                               [0,0,1,0],
#                               [0,0,0,1],
#                               [0,0,0,0]))
Connection_neuro = 'Random'

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
    N,S = Neuronal_Network(Nn,I_inj,Connection_neuro,sigma_noise,sed)
    

    
    
    
    
    
    
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
    N,S = Neuronal_Network(Nn,I_inj,Connection_neuro,sigma_noise,sed)
   
    

    
    
    
elif Simulated_network == 'Astrocytic':   
 
    '''
    Glutamate stimulation is delivered to randomly choosen astrocytes in number 
    N_stim in the parameters.
    '''
    # --------- ASTROCYTE -----------
    Astro, GJ,P,Glu_Input = Astrocyte_Group(Na,Connection_astro,Simulated_network,sed)
    



# ------------------------- NETWORK SIMULATION -------------------------

# --- Monitors ---
# String for recording the time course of neuron parameters
recording_stringN = ['v', 'u', 'I', 'I_syn','I_ampa','I_nmda','x_d']
# recording_stringS = ['x_d']


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


#%%
# --------------------- PLOTS ---------------------
# ------- NEURONS -------
%matplotlib
fig, (ax1, ax2, ax3,ax4) = plt.subplots(4, 1) # Added figsize for better viewing



ax1.plot(MonitorN.t / second, MonitorN[0].v , 'k', linewidth=0.7)
ax1.set_ylabel('Voltage [mV]')

ax2.plot(MonitorN.t / second, MonitorN[1].v , 'k', linewidth=0.7)
ax2.set_ylabel('Voltage [mV]')



ax3.plot(MonitorN.t / second, MonitorN[2].v , 'k', linewidth=0.7)
ax3.set_ylabel('Voltage [mV]')



ax4.plot(MonitorN.t / second, MonitorN[3].v , 'k', linewidth=0.7)
ax4.set_xlabel('Time [s]')
ax4.set_ylabel('Voltage [mV]')
fig.show()

plt.figure(dpi=200)
plt.plot(SpikesN.t / second, SpikesN.i, '.k', ms=0.7)

show()



fig, (ax1, ax2, ax3,ax4) = plt.subplots(4, 1) # Added figsize for better viewing



ax1.plot(MonitorN.t / second, MonitorN[0].u , 'k', linewidth=0.7)
# ax1.set_ylabel('')

ax2.plot(MonitorN.t / second, MonitorN[1].u , 'k', linewidth=0.7)
# ax2.set_ylabel()



ax3.plot(MonitorN.t / second, MonitorN[2].u , 'k', linewidth=0.7)
# ax3.set_ylabel('Voltage [mV]')



ax4.plot(MonitorN.t / second, MonitorN[3].u , 'k', linewidth=0.7)
ax4.set_xlabel('Time [s]')
# ax4.set_ylabel('u variable [mV]')
fig.show()

plt.figure(dpi=200)
plt.plot(SpikesN.t / second, SpikesN.i, '.k', ms=0.7)

show()

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




fig, (ax1, ax2, ax3,ax4) = plt.subplots(4, 1) # Added figsize for better viewing



ax1.plot(MonitorN.t / second, MonitorN[0].x_d , 'k', linewidth=0.7)
ax1.set_ylabel('x_d')

ax2.plot(MonitorN.t / second, MonitorN[1].x_d , 'k', linewidth=0.7)
ax2.set_ylabel('x_d')



ax3.plot(MonitorN.t / second, MonitorN[2].x_d , 'k', linewidth=0.7)
ax3.set_ylabel('x_d')



ax4.plot(MonitorN.t / second, MonitorN[3].x_d , 'k', linewidth=0.7)
ax4.set_xlabel('Time [s]')
ax4.set_ylabel('x_d')
