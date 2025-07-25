
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

def Neuronal_Network(Adaptation,I_inj):
    
# std_pers = persentage of mean value used as standard deviation for introducing some
#   variability
# ---------------------- NEURONAL GROUP ----------------------
     # neuron model
    eqs_NN = Equations('''
    
    dV/dt = noise + (-gl*(V-El)-g_na*(m*m*m)*h*(V-ENa)-g_kd*(n*n*n*n)*(V-EK)-g_AHP*p*(V-EK)+I-I_syn)/Cm  : volt
    dm/dt = alpha_m*(1-m)-beta_m*m : 1
    dh/dt = alpha_h*(1-h)-beta_h*h : 1
    dn/dt = (alpha_n*(1-n)-beta_n*n) : 1
    dp/dt = (p_ss - p)/tau_p : 1
    # dhp/dt = 0.128*exp((17.*mV-V+VT)/(18.*mV))/ms*(1.-hp)-4./(1+exp((30.*mV-V+VT)/(5.*mV)))/ms*h : 1
    alpha_m = 0.32*(mV**-1)*4*mV/exprel((13*mV-V+VT)/(4*mV))/ms : Hz
    beta_m = 0.28*(mV**-1)*5*mV/exprel((V-VT-40*mV)/(5*mV))/ms : Hz
    alpha_h = 0.128*exp((17*mV-V+VT)/(18*mV))/ms : Hz
    beta_h = 4./(1+exp((40*mV-V+VT)/(5*mV)))/ms : Hz
    alpha_n = 0.032*(mV**-1)*5*mV/exprel((15*mV-V+VT)/(5*mV))/ms : Hz
    beta_n = .5*exp((10*mV-V+VT)/(40*mV))/ms : Hz
    p_ss = (1./(exp(-(V + 35*mV)/(10*mV))+1)) : 1
    tau_p = Tau_max / (3.3*exp( (V + 35*mV)/(20*mV) ) + exp( -( V + 35*mV )/(20*mV) )) : second
    
    noise = sigma*(2*gl/Cm)**.5*randn()/sqrt(dt) : volt/second (constant over dt)
    I : amp
    I_syn : amp
    I_AHP = g_AHP*p*(V-EK) : volt * siemens 
    x : meter
    y : meter
    ''')
    



    
    # -------------------------- INITIALIZE THE NETWORKS ---------------------------
    
    # ---- Get parameters ----
    params_NN = get_Neuronparam(Adaptation,I_inj=I_inj)
    
    
    P = NeuronGroup(1, model=eqs_NN, name='Neuron*',namespace= params_NN, threshold='V>20*mV', refractory=2 * ms,
                        method='exponential_euler')
    
    # Initialize neuron parameters
    P.V = -39 * mV                          # approximately resting membrane potential
    P.I = 'I_inj'          # Make neurons heterogeneously excitable
    
    
 
    return P
        
def get_Neuronparam(Adaptation=False,delta = 0,**kwargs):
    
    
    Neuron_area =  300*umetre**2
    
    if Adaptation == True:
            
            g_AHP= (0.1*msiemens*cm**-2) * Neuron_area
            
            
    else:
            g_AHP = 0

    
    
    
    params = { # --- Neuron Parameters
    'area': Neuron_area,               # membrane area of the neuron
    'Cm': (2*ufarad*cm**-2) * Neuron_area, # membrane capacitance (calculated with area)
    'El': -39.2 * mV,                    # Nernst potential of leaky ions
    'EK': -80 * mV,                      # Nernst potential of potassium
    'ENa': 70 * mV,                      # Nernst potential of sodium
    'g_na': 1.6 * 50 * msiemens * cm**-2 * Neuron_area, # maximal conductance of sodium channels (calculated with area)
    'g_kd': 1.3 * 5 * msiemens * cm**-2 * Neuron_area,  # maximal conductance of potassium (calculated with area)
    'gl': (0.3*msiemens*cm**-2) * Neuron_area, # maximal leak conductance (calculated with area)
    'g_AHP': g_AHP, # maximal conductance of AHP currents
    'VT': -30.4*mV,                      # alters firing threshold of neurons
    'sigma': 6 * mV,                     # standard deviation of the noisy voltage fluctuations
    'Tau_max': 4000 * ms,                # Decay factor of AHP
    
    'I_inj': 15*pA, # Injected current
 
     # Synaptic contribution
     'we_AMPA' : 0.5, # Relative contribution of AMPA channels to the total syn weight
     'we_NMDA' : 0.5, # Relative contribution of NMDA channels to the total syn weight
     'g_ampa': (1 + delta) * nS, # Note: if delta is a direct value, not a key lookup, it should be 0.6
     'g_nmda': (1 - delta) * nS, # Note: if delta is a direct value, not a key lookup, it should be 0.6
     'E_ampa': 0 * mV,
     'E_nmda': 0 * mV,
    
 
    
 
    # Adaptation parameters (uncommented and added to dictionary)
    # If these are meant to be included, they should also be added as key-value pairs
    # 'E_AHP': EK, # Note: if EK is a direct value, not a key lookup, it should be -80*mV
    # 'g_AHP': 5 * nS,
    # 'tau_Ca': 8000 * ms,
    # 'alpha_Ca': 0.00035,
 
    # Synapse parameters (uncommented and added to dictionary)
    # If these are meant to be included, they should also be added as key-value pairs
    # 'S': 0.4,
    # 'delta': 0.6,
    # 'g_ampa': (1 + delta) * nS, # Note: if delta is a direct value, not a key lookup, it should be 0.6
    # 'g_nmda': (1 - delta) * nS, # Note: if delta is a direct value, not a key lookup, it should be 0.6
    # 'E_ampa': 0 * mV,
    # 'E_nmda': 0 * mV,
    # 'tau_ampa': 2 * ms,
    # 'taus_nmda': 100 * ms,
    # 'taux_nmda': 2 * ms,
    # 'alpha_nmda': 0.5 * kHz,
    # 'tau_d': 200 * ms,
    # 'U': 0.2,
    # 'STF': False,
    # 'tau_f': 1000 * ms,
 
    # Asynchronous Release parameters (uncommented and added to dictionary)
    'AsynchronousRelease': False,
    'tau_ar': 700 * ms,
    'Uar': 0.003,
    'Umax': 0.5/ms,
    'x0': 5, # x0 seems to be unitless here

     }
    params.update(kwargs)

    return params

# --------- SIMULATION -----------
simtime = 2 * second               # simulation time
# transient = 3 * second              # time omitted as transient
sed = 39                             # random number seed
devices.device.seed(sed)            # set the seed for all the random number realisations

Adaptation = True
I_inj = 30*pA


N = Neuronal_Network(Adaptation,I_inj)



recording_stringN = ['V','I_AHP']
MonitorN = StateMonitor(N, recording_stringN, record=True)


net_ = Network(collect())  # automatically include all the stated groups
net_.run(simtime,report='text', profile=True)

#%%
%matplotlib

plt.plot()
plt.plot(MonitorN.t / second, MonitorN[0].V / mV, 'k', linewidth=0.7)
plt.ylabel('Voltage [mV]')

plt.show()
