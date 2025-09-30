"""
Created on Tue Aug 12 17:53:15 2025

@author: Admin
"""


import matplotlib.pyplot as plt
from brian2 import *

from brian2 import clear_cache


import os


os.chdir(r'C:\Users\Admin\Desktop\Leonardo\ASN')
from ASN_fun import *


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
    2) Only the TM_coupled model is used for synaptic weight modulation
    3) Asynchronous release targets both ampa and nmda receptors
    4) The synaptic model scales r_nmda and r_ampa by an alpha factor to match the currents observed
    5) Autapses are not modeled.
    6) For Glia-Syn connections the rule is distance based as in [2]. The synapse position is located at the post-synaptic 
        neuron given the discrepances in the extention of axonal and dendritic domains



    
    



REFERENCES:
    1) Distance dependent prob of conneciton = Estimating neuronal connectivity from axonal and dendritic density fields
    2) Astrocyte connections: A Computational Model of Interactions Between Neuronal and Astrocytic Networks: The Role of Astrocytes in the Stability of the Neuronal Firing Rate

    
    
    
TODO:
    1) Syn to astro and astro to syn functions must be improved by defining a distance dependent connections.
    2) Add ics in the astrocytes
'''





# Saving 
Out_path = r'C:\Users\Admin\Desktop\Leonardo\ASN\Output Temp'


BrianLogger.suppress_hierarchy('brian2.devices')
BrianLogger.suppress_hierarchy('brian2.parsing')


# Clear the cache for the 'cython' code generation target
# clear_cache('cython') 
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

Asynchronous_release = True

# ------- Neurons -------
Adaptation = True


# ------- Astrocytes -------
oscillations = 'AM'










# ------------------------- PARAMETERS -------------------------

# --------- SIMULATION -----------
simtime =50 * second               # simulation time
# transient = 3 * second              # time omitted as transient
sed_neuron = 39                             # random number seed
sed_astro= 60
devices.device.seed(sed)            # set the seed for all the random number realisations


Simulated_network = 'Full' # Astrocytic/Neuronal/Full


# --------- NEURON -----------
Nn =100
neuron_radius = 9 #[um]
# --------- SYNAPTIC -----------

# ----- Connectivity -----
# Can be directly an ADJ or a string: Random, Distance
# Connection_neuro = np.array(([0,1,0,0],
                              # [0,0,1,0],
                              # [0,0,0,1],
                              # [0,0,0,0]))
Connection_neuro = 'Random'
conn_prob_ = 0.13

# --------- ASTROCYTE and GJ ----------- 
'''
Given the nature of the link hte adjency matrix is ALWAYS symmetric

'''
Na = 40

# ----- Connectivity -----
# Can be directly an ADJ or a string: Distance
# Connection_astro= np.array(([0,1,0],
#                             [1,0,1],
#                             [0,1,0]))
Connection_astro = 'Distance'


# --------- GLIOTRANSMISSION ----------- 


# --------- ASTRO-NEURON LINKS ----------- 
# --- Synapse to astro ---
# ADJ_SynAstro = np.array(([1,0,0],
#                          [0,1,0],
#                          [0,0,1]))
Connection_StoA = 'Distance'


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
    N,S = Neuronal_Network(Nn,Connection_var = 'Random',
                        add_delay= False,delay_mode= 'random',
                         Max_Delay = 10*ms,ics = False, Simulated_network = Simulated_network,
                         Decay_type = 'Double_exp',synapse_type = 'facilitating', conn_prob_ = conn_prob_,sed=sed_neuron)
    
    

    N.I = '(rand() -0.5) * I_inj'          # Make neurons heterogeneously excitable
    
    
    
    
    
    # --------- ASTROCYTE -----------
    Astro,GJ = Astrocyte_Group(Na,Connection_astro,Simulated_network,sed_astro)
    
    
    
    
    # --------- GLIOTRANSMISSION ----------- 
    GT = Gliotransmission(Na,ics,Astro)
    
    
    # --------- ASTRO-NEURON LINKS ----------- 
    # --- Synapse to astro ---
    StoA,Source_syn,Target_astro  = Synapse_to_astro(S,Astro,Connection_StoA)
    
    # --- Astro to synapse ---
    AtoS = Astro_to_Syn(GT,S,Target_astro,Source_syn)   # Astro_to_Syn(Glio_release,synapse,Source_astro, Target_syn)
    
    
    
    
elif Simulated_network == 'Neuronal':

    # --------- NEURON and SYNAPSE -----------
    N,S = Neuronal_Network(Nn,Connection_var = 'Random',
                        add_delay= False,delay_mode= 'random',
                         Max_Delay = 10*ms,ics = False, Simulated_network = 'Neuronal',
                         Decay_type = 'Double_exp',synapse_type = 'facilitating', conn_prob_ = conn_prob_,sed=sed)
    # N.namespace['sigma']=4.1*mV
    # N.namespace['I_inj']=0*pA
    N.I = '(rand() -0.5) * I_inj'          # Make neurons heterogeneously excitable
   
    

    
    
    
elif Simulated_network == 'Astrocytic':   
 
    '''
    Glutamate stimulation is delivered to randomly choosen astrocytes in number 
    N_stim in the parameters.
    '''
    # --------- ASTROCYTE -----------
    Astro, GJ,P,Glu_Input = Astrocyte_Group(Na,Connection_astro,Simulated_network,sed)
    



# ------------------------- NETWORK SIMULATION -------------------------

# --- Monitors ---
recording_stringN = ['V','I_syn','I_ampa','I_nmda','I_cell']
# recording_stringN = ['V','I_ampa','I_nmda','I_AHP']
recording_stringS = ['usr','x_S','Y_S','uar','r_Sr','r_Ar','r_ampa','r_nmda']
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
    MonitorS = StateMonitor(S, recording_stringS, record=True)
    MonitorN = StateMonitor(N, recording_stringN, record=True)
    SpikesN = SpikeMonitor(N)
    
    
elif Simulated_network == 'Astrocytic': 
    MonitorG = StateMonitor(Glu_Input, ['Y_bias_in'], record=True)
    MonitorA = StateMonitor(Astro, recording_stringA, record=True)
    SpikesA = SpikeMonitor(Astro)
    SpikesP = SpikeMonitor(P)
    
    
#
%matplotlib
plot_connections(N, Astro, S, StoA)
# --- Collect and add monitors ---
#%%

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





















#%%
fig.show()

plt.figure(dpi=200)
plt.plot(SpikesN.t / second, SpikesN.i, '.k', ms=0.69)

show()

#%%
# CHECK AR

fig, (ax1, ax2, ax3) = plt.subplots(3, 1) # Added figsize for better viewing


# ax1.plot(SpikesN.t/second,spike_trains[0],'.g', ms=5,label='Spikes')
ax1.plot(MonitorS.t / second, MonitorS[0].uar/hertz, 'r', linewidth=0.7,label='uar')

ax1.plot(MonitorS.t / second, MonitorS1[0].uar/hertz,'--', linewidth=0.7,label='No uar')


ax1.legend()


# ax2.plot(SpikesN.t/second,spike_trains[1],'.g', ms=5,label='Spikes')
ax2.plot(MonitorS.t / second, MonitorS[1].uar/hertz, 'r', linewidth=0.7)

ax2.plot(MonitorS.t / second, MonitorS1[1].uar/hertz, '--', linewidth=0.7)



# ax3.plot(SpikesN.t/second,spike_trains[2],'.g', ms=5,label='Spikes')
ax3.plot(MonitorS.t / second, MonitorS[2].uar/hertz, 'r', linewidth=0.7)

ax3.plot(MonitorS.t / second, MonitorS1[2].uar/hertz,'--',  linewidth=0.7)

ax3.set_xlabel('Time [s]')
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
ax2.plot(MonitorS.t / second, MonitorS[1].r_ampa/hertz, 'r', linewidth=0.7)

ax2.plot(MonitorS.t / second, MonitorS[1].r_nmda/hertz, '--', linewidth=0.7)



# ax3.plot(SpikesN.t/second,spike_trains[2],'.g', ms=5,label='Spikes')
ax3.plot(MonitorS.t / second, MonitorS[2].r_ampa/hertz, 'r', linewidth=0.7)

ax3.plot(MonitorS.t / second, MonitorS[2].r_nmda/hertz,'--',  linewidth=0.7)

ax3.set_xlabel('Time [s]')


fig, (ax1, ax2, ax3) = plt.subplots(3, 1) # Added figsize for better viewing


# ax1.plot(SpikesN.t/second,spike_trains[0],'.g', ms=5,label='Spikes')
ax1.plot(MonitorS.t / second, MonitorS[0].r_Sr/hertz, 'r', linewidth=0.7,label='r_Sr')

ax1.plot(MonitorS.t / second, MonitorS[0].r_Ar/hertz,'--', linewidth=0.7,label='r_Ar')


ax1.legend()


# ax2.plot(SpikesN.t/second,spike_trains[1],'.g', ms=5,label='Spikes')
ax2.plot(MonitorS.t / second, MonitorS[1].r_Sr/hertz, 'r', linewidth=0.7)

ax2.plot(MonitorS.t / second, MonitorS[1].r_Ar/hertz, '--', linewidth=0.7)



# ax3.plot(SpikesN.t/second,spike_trains[2],'.g', ms=5,label='Spikes')
ax3.plot(MonitorS.t / second, MonitorS[2].r_Sr/hertz, 'r', linewidth=0.7)

ax3.plot(MonitorS.t / second, MonitorS[2].r_Ar/hertz,'--',  linewidth=0.7)

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
ax3.plot(MonitorS.t / second, MonitorS[954].usr, 'r', linewidth=0.7)

ax3.plot(MonitorS.t / second, MonitorS[954].x_S, 'c', linewidth=0.7)

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


