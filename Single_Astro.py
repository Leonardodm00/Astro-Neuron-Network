# -*- coding: utf-8 -*-
"""
Created on Thu Jul 24 15:10:10 2025

@author: Admin
"""

# -*- coding: utf-8 -*-
"""
Created on Thu Jul 24 12:46:25 2025

@author: leona
"""
import matplotlib as mpl
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error
import numpy as np
# import torch
import scipy.io
import pdb
import os


from brian2 import *
from brian2 import devices








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
        'C_Theta': 0.5*umole,      # Ca^2+ threshold for exocytosis
        'Omega_A': 0.6/second,     # Gliotransmitter recycling rate
        'U_A': 0.6,                # Gliotransmitter release probability
        'G_T': 200.*mmole,         # Total vesicular gliotransmitter
        'rho_e': 6.5e-4,           # Ratio of astrocytic vesicle volume/ESS volume
        'Omega_e': 5./second,      # Gliotransmitter clearance rate (think about distributed release)
        'spill_over': 0.75,         # Spill over parameter
      
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
    


def Astrocyte_Group(N_astro=1,ics=None):
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
        x : meter
        y : meter
        ''')
       
       
       
    Params_astroGT = get_Astroparam()
       
    # The definition of a threshold and reset mechanism in the astrocyte group
    # allows to use SpikeMonitors to estimate the frequency of oscillations
    Astro = NeuronGroup(N_astro, eqs_A,
                        threshold='C>C_osc',
                        refractory='C>C_osc',
                        method='rk4',
                        namespace=Params_astroGT,
                        name='astrocyte*')
    
    # Random initialization of initial conditions
    if ics=='rand':
        Astro.Gamma_A = 'rand()'
        Astro.I = '3*rand()*umole'
        Astro.C = '1.5*rand()*umole'
        Astro.h = 'rand()'
        
        
        
        
    # ----- Gap-junction based astro links -----
    
    #####################   NOT NEEDED  ############################
    # Gap_Eq = Equations('''
                     
            
    #                     delta_I = I_post - I_pre: mole
    #                     I_coupling = -F/2*(1 + tanh((abs(delta_I) - I_Theta)/omega_I))*sign(delta_I) : mole/second 
    #                     I_coupling_tot_post = I_coupling : mole/second (summed)
                        
                       
    #                     ''')
    
    # GJ = Synapses(Astro,Astro,
    #               model=Gap_Eq,
    #               method='exponential_euler',
    #               namespace= Params_astroGT,
    #               name = 'Gap_junctions*'
    #               )
    
    
    
    
    # # ----- Connections -----
    
    # Source_astro,Target_astro = unfold_ADJ(ADJ)
    
    # GJ.connect(i = Source_astro,j= Target_astro)
    ####################################################################################
    
    
    # ---------- EXTERNAL STIMULATION ----------
    Params_astroGT.update({'tau_glustim' :25*ms}) # As for synapse params
    Params_astroGT.update({'Y_bias_max' : 1*mmole}) # Maximum glutamate concentration
    Params_astroGT.update({'poisson_rate' : 2*Hz}) # Lambda parameter of the poisson process
    
    
    # --- Poisson process ---
    P = PoissonGroup(1, Params_astroGT['poisson_rate'])
    
    
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

    Glu_Input.connect(i=0, j=0)
    
    
        
        
        
        
    return Astro,  Glu_Input,P





start_scope()
simtime = 100 *second
sed = 39                            # random number seed
devices.device.seed(sed)            # set the seed for all the random number realisations





Astro,  Glu_Input,P = Astrocyte_Group()

recording_stringGLU = ['Y_bias']
recording_stringA = ['C','I','Gamma_A']

MonitorA = StateMonitor(Astro, recording_stringA, record=True)
MonitorG = StateMonitor(Glu_Input, recording_stringGLU, record=True)
SpikesP = SpikeMonitor(P)
SpikesA = SpikeMonitor(Astro)
net_ = Network(collect())  # automatically include all the stated groups
net_.run(simtime,report='text', profile=True)

#%%
%matplotlib
plt.figure()
maxC = np.max( MonitorA[0].C)
maxI = np.max( MonitorA[0].I)
plt.plot(MonitorA.t/second, MonitorA[0].C/maxC, 'k', label='Calcium')
plt.plot(MonitorA.t/second, MonitorA[0].I/maxI, 'r', label='IP3')
plt.plot(MonitorA.t/second, MonitorA[0].Gamma_A, 'b', label='Frac bounded GPCR')
# ax1.plot(SpikesN.t/second, SpikesN[0].i+1e-11, '.g', ms=5, label='Spikes') # Or a more descriptive label like 'Neuron SpikesN'

plt.legend()


plt.figure()
plt.plot(MonitorG.t/second, MonitorG[0].Y_bias, 'k', label='Calcium')


plt.legend()

plt.figure()
plt.plot(SpikesP.t/second, SpikesP.i, '.k', ms=5)

show()

plt.figure()
plt.plot(SpikesA.t/second, SpikesA.i, '.k', ms=5)

show()
