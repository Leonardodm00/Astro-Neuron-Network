# -*- coding: utf-8 -*-
"""
Created on Sun Jul 20 11:01:09 2025

@author: paolo
"""

import matplotlib.pyplot as plt
from brian2 import *


import os


os.chdir(r'C:\Users\paolo\Desktop\PYTHON DATA')
import ANS_fun


'''


The phantom network is defined as follows












'''












start_scope()
# ------------------------- SET OPTIONS -------------------------

# ------- Synapses -------
synapse='neutral'
ics=None 
dt=None            

postc_sic='double-exp'
sic=None 
delay=None
RandomKinetics = False 
OnlyExc = True
std_pers =0.01

Syn_Currents_model = 'Nina' # 'Kinetic','Nina

Max_delay = 25 *ms
add_delay = False
delay_mode = 'random'


# ------- Neurons -------
Adaptation = True


# ------- Astrocytes -------
oscillations = 'AM'










# ------------------------- PARAMETERS -------------------------

# --------- SIMULATION -----------
simtime = 100 * second               # simulation time
transient = 3 * second              # time omitted as transient
sed = 39                             # random number seed
devices.device.seed(sed)            # set the seed for all the random number realisations
dt = defaultclock.dt 

Simulated_network = 'Neuronal' # Astrocytic/Neuronal/Full


# --------- NEURON -----------
Nn = 4


# --------- SYNAPTIC -----------
# ----- Connectivity -----
ADJ_neuro


# --------- ASTROCYTE and GJ ----------- 
Na = 3
ADJ_astro = 


# --------- GLIOTRANSMISSION ----------- 


# --------- ASTRO-NEURON LINKS ----------- 
# --- Synapse to astro ---
ADJ_SynAstro = 


# --- Astro to synapse ---
ADJ_AstroSyn = 









# ------------------------- GROUPS BUILD-UP -------------------------

if Simulated_network == 'Full':
    # --------- NEURON and SYNAPSE -----------
    N,S = Neuronal_Network(Nn,Source_neuron,Target_neuron, RandomKinetics, OnlyExc,
                           Syn_Currents_model,add_delay,delay_mode,Max_delay,ics,
                           std_pers, Simulated_network)
    
    
    # --------- ASTROCYTE -----------
    A,GJ = Astrocyte_Group(N_astro,Source_astro,Target_astro)
    
    
    # --------- GLIOTRANSMISSION ----------- 
    GT = Gliotransmission(N_astro,ics,Astro)
    
    
    # --------- ASTRO-NEURON LINKS ----------- 
    # --- Synapse to astro ---
    StoA = Synapse_to_astro(S,A,Pre_syn,Post_astro)
    
    # --- Astro to synapse ---
    AtoS = Astro_to_Syn(GT,S,Pre_astro,Post_syn)
    
    
    
    
elif Simulated_network == 'Neuronal':

    # --------- NEURON and SYNAPSE -----------
    N,S = Neuronal_Network(Nn,Source_neuron,Target_neuron, RandomKinetics, OnlyExc,
                           Syn_Currents_model,add_delay,delay_mode,Max_delay,ics,
                           std_pers, Simulated_network)
    
    
    
elif Simulated_network == 'Atrocytic':   
    
    # --------- ASTROCYTE -----------
    A,GJ = Astrocyte_Group(N_astro,Source_astro,Target_astro)
    
    
    
    
    






# ------------------------- NETWORK SIMULATION -------------------------

# --- Monitors ---
recording_stringP = ['V']
recording_stringS = ['u_S','x_S','Y_S','I_syn','I_ampa','I_nmda']
recording_stringA = ['C','I','Gamma_A','I_coupling_tot']
recording_stringGT = ['G_A','x_A']


# collect

