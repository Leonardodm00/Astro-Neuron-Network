
'''

Post-synaptic currents fitting procedure with Bayesian Optimization algorithm.

TK synaptic model and Glutamate driven synaptic activity with simplyfied first order
kinetic model.



Parameters' name:
    
    Param[0]: Y_T = Total vescicular glutamate concentration
    Param[1]: alpha = Binding rate
    Param[2]: beta = Dissociation rate


Main steps:
    

- First random search with full range sizes;
- Second random search: unimportant features are remuved and the remaining ones' 
                ranges are narrowed within the 10% of the best values;
- Choose median or best HP values but the ridge;




Searching algorithm: Bayesian Optimization by Gaussian Process.

This pipeline doesn't require the use K-fold cross validation.   

Evaluation metrics are Normalized Root Mean Squared Error and R^2 correlation coefficient.

REFERENCES:

    Bayesian Optimization algorithm: Scikit-optimize library.
   
    
   
NOTES:
    
    - Remember to change the alpha and beta target, i.e. if of ampa, nmda or gaba receptors.
    - Remember to change the Monitor and fitted curve
    
    
    
    

'''


from skopt.space import Real, Integer
import matplotlib as mpl
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error
import numpy as np
# import torch
import scipy.io
from scipy.signal import find_peaks
import pdb
import os
import pandas as pd


from brian2 import *
from brian2 import devices
from brian2.core.functions import timestep


def Synapse_simulation(Params):
    
    
    '''
    Syynapse wrapper: Brian2 based simulation
    
 
    
    '''

    # ----------- Load and preprocess data -----------
    # Upload the reference data that we're trying to fit.
    
    # Load data
    os.chdir(r'C:\Users\leona\Desktop\Temp scripts\CURRENTS')
    with open('I_AMPA.csv', 'r') as file:
        Ref_data = file.read()
    
    
    # ---------- Initial params -----------
    start_scope()


    # simulation parameters
    simtime = 6 * second               # simulation time
    sed = 39                             # random number seed
    devices.device.seed(sed)            # set the seed for all the random number realisations
   
    
    # Tune the trace extracion window
    Window_length = 300 * ms
    Start_time = 1 * second
    
    # ---- Paramter extraction ----
    
    Y_T_t = Params[0]
    alpha_t = Params[1]
    beta_t = Params[2]
    
    # ----------------- NEURONAL EQUATIONS ----------------------
    
    
     # neuron model
    eqs_NN = Equations('''
    # dV/dt = noise + ((-gl*(V-El)-g_na*(m*m*m)*h*(V-ENa)-g_kd*(n*n*n*n)*(V-EK)+I+I_AHP)/Cm) : volt
    dV/dt = noise + (-gl*(V-El)-g_na*(m*m*m)*h*(V-ENa)-g_kd*(n*n*n*n)*(V-EK)+I-I_syn)/Cm  : volt
    dm/dt = alpha_m*(1-m)-beta_m*m : 1
    dh/dt = alpha_h*(1-h)-beta_h*h : 1
    dn/dt = (alpha_n*(1-n)-beta_n*n) : 1
    dhp/dt = 0.128*exp((17.*mV-V+VT)/(18.*mV))/ms*(1.-hp)-4./(1+exp((30.*mV-V+VT)/(5.*mV)))/ms*h : 1
    alpha_m = 0.32*(mV**-1)*4*mV/exprel((13*mV-V+VT)/(4*mV))/ms : Hz
    beta_m = 0.28*(mV**-1)*5*mV/exprel((V-VT-40*mV)/(5*mV))/ms : Hz
    alpha_h = 0.128*exp((17*mV-V+VT)/(18*mV))/ms : Hz
    beta_h = 4./(1+exp((40*mV-V+VT)/(5*mV)))/ms : Hz
    alpha_n = 0.032*(mV**-1)*5*mV/exprel((15*mV-V+VT)/(5*mV))/ms : Hz
    beta_n = .5*exp((10*mV-V+VT)/(40*mV))/ms : Hz
    noise = sigma*(2*gl/Cm)**.5*randn()/sqrt(dt) : volt/second (constant over dt)
    I : amp
    
    
    x : meter
    y : meter
    ''')
    
    
    
    
    
    # ----------------- SYNAPTIC EQUATIONS ----------------------
    
    # Basic synaptic equations are the same. what changes is how the post synaptic current behaves.
    
  
    
    eqs_Syn = Equations('''
    # Fraction of activated presynaptic receptors
    
    
    # Usage of releasable neurotransmitter per single action potential:
    du_S/dt = -Omega_f * u_S : 1 (clock-driven)
    
    # Fraction of synaptic neurotransmitter resources available for release:
    dx_S/dt = Omega_d *(1 - x_S) : 1 (clock-driven)
    dY_S/dt = -Omega_c * Y_S : mole (clock-driven)
    
    
    # Define the variables of the model
    G_A : mole  # gliotransmitter concentration in the extracellular space
    U_0 : 1
    r_S : 1     # Because r_S is the product of u_S and x_S that are event-driven, it is itself event-driven too
    
    
     
    # Astrocyte ID for connection
    astro_index : integer
    # Per-synapse gliotransmitter-effect parameter
    alpha  : 1
    ''')
    
    # -------------- Event based update --------------
    
    
    pre = '''
    
    U_0 =  U_0__star
    u_S += U_0 * (1 - u_S)
    r_S = u_S * x_S # released synaptic neurotransmitter resources
    x_S -= r_S
    Y_S += rho * Y_T * r_S
    '''
    post = None
    

    
    # ----------- UPGRADED MODEL -------------
    
    
    eqs_Syn += Equations('''
                         
                         
                            dr_ampa/dt = alpha_ampa_new * Y_S * (1 - r_ampa) - beta_ampa_new * r_ampa: 1 (clock-driven)
                            dr_nmda/dt =  alpha_nmda_new * Y_S * (1 - r_nmda) - beta_nmda_new * r_nmda : 1 (clock-driven)
                            
              
                            
                            r_ampa_tot_post = r_ampa : 1 (summed)
                            r_nmda_tot_post = r_nmda : 1 (summed)
                         
                         
                        ''')
                        

    
    eqs_NN += Equations(''' 
                        I_syn =  I_ampa + I_nmda: amp
                        I_ampa = g_ampa*(V-E_ampa)*(r_ampa_tot) : amp
                        I_nmda = g_nmda*(V-E_nmda)*(r_nmda_tot)/(1+exp(-0.062*V/mV)/3.57) : amp
                        r_nmda_tot :1
                        r_ampa_tot :1
                        
                    
                        ''')
    
    
    
    
    
    # ----------- Network build -----------
    
    params_NN = get_Neuronparam(sigma = 0*mV)
    params_Syn = get_Synparam(synapse = 'neutral', Y_T = Y_T_t *mmole, alpha_ampa_new = alpha_t * 1/mole * 1/second, beta_ampa_new = beta_t * 1/second)
   

    
    
    poisson_rate = 1*Hz
    P = PoissonGroup(1, poisson_rate)
    
    neuron = NeuronGroup(1, eqs_NN, threshold='V > 0*mV', refractory=2*ms,
                     method='exponential_euler',namespace=params_NN)
    neuron.V = -39 *mV # Initialize neuron voltage
    
    
    synapse = Synapses(P, neuron, model=eqs_Syn,
                   on_pre=pre, on_post=post,namespace=params_Syn,method='exponential_euler')
    
    
    #%
    # Connect the input spikes to the neuron
    synapse.connect(i=0, j=0) # Connect the single input "neuron" to the single target neuron
    synapse.x_S = 1.0
    # --- Monitoring ---
    state_monitor = StateMonitor(neuron, ['V','I_syn','I_ampa','I_nmda'], record=0)
    # state_monitorsynapse = StateMonitor(synapse, ['Y_S','r_ampa','r_nmda'], record=0)
    spike_monitor_neuron = SpikeMonitor(neuron)
    spike_monitor_input = SpikeMonitor(P)
    
   
    # --- Run Simulation ---
    run(simtime)
     
     
    # --- Extract trace ---
    
    Simulated_trace = state_monitor[0].I_ampa

    # ---------- Retrun the  NMRE ----------
         
        
    return Simulated_trace,Ref_data






def get_Neuronparam(Adaptation=False,delta = 0.5,**kwargs):
    
    
    Neuron_area =  300*umetre**2
    
    if Adaptation == True:
            
            g_m= (0.4*msiemens*cm**-2) * Neuron_area
            
            
    else:
            g_m = 0

    
    
    
    params = { # --- Neuron Parameters
    'area': Neuron_area,               # membrane area of the neuron
    'Cm': (2*ufarad*cm**-2) * Neuron_area, # membrane capacitance (calculated with area)
    'El': -39.2 * mV,                    # Nernst potential of leaky ions
    'EK': -80 * mV,                      # Nernst potential of potassium
    'ENa': 70 * mV,                      # Nernst potential of sodium
    'g_na': 1.6 * 50 * msiemens * cm**-2 * Neuron_area, # maximal conductance of sodium channels (calculated with area)
    'g_kd': 1.3 * 5 * msiemens * cm**-2 * Neuron_area,  # maximal conductance of potassium (calculated with area)
    'gl': (0.3*msiemens*cm**-2) * Neuron_area, # maximal leak conductance (calculated with area)
    'g_m': g_m, # maximal conductance of AHP currents
    'VT': -30.4*mV,                      # alters firing threshold of neurons
    'sigma': 6 * mV,                     # standard deviation of the noisy voltage fluctuations
    'Tau_max': 4000 * ms,                # Decay factor of AHP
 
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






def get_Synparam(synapse='depressing',**kwargs):
    
        
        
    params = {
        
        'area' : 300*umetre**2,  
        # --- Synaptic dynamics
        'E_Iper': 80/100,          # Persentage of excitatory connections
        # Omega_d (see below)      # Depression rate
        # Omega_f (see below)      # Facilitation rate,
        # U_0__star (see below)    # Basal synaptic release probability
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
        
        
       
       # Adaptation parameters (uncommented and added to dictionary)
       # If these are meant to be included, they should also be added as key-value pairs
        'E_AHP': -80 * mV, # Note: if EK is a direct value, not a key lookup, it should be -80*mV
        'g_AHP': 5 * nS,
        'tau_Ca': 8000 * ms,
        'alpha_Ca': 0.00035,
    
       # Synapse parameters (uncommented and added to dictionary)
       # If these are meant to be included, they should also be added as key-value pairs
        'S': 0.4,
        'delta': 0.6,
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
        
        # Params of the upgrated model
        'alpha_ampa_new' : 1.1e6 * 1/mole * 1/second,
        'alpha_nmda_new' : 7.2e4 * 1/mole * 1/second,
        'beta_ampa_new' : 190 * 1/second,
        'beta_nmda_new' :  6.6 * 1/second,
        'epsilon': 1e-40 * Hz,
        
       
    
       # Asynchronous Release parameters (uncommented and added to dictionary)
       'AsynchronousRelease': False,
       'tau_ar': 700 * ms,
       'Uar': 0.003,
       'Umax': 0.5/ms,
       'x0': 5, # x0 seems to be unitless here
    
    }
    
    
    # ------------------ SYNAPSES ------------------
    if synapse == 'depressing':
        params.update({
            'Omega_d': 2./second,
            'Omega_f': 3.33/second,
            'U_0__star': 0.6,
            'alpha': 0.,
        })
    elif synapse == 'facilitating':
        params.update({
            'Omega_d': 2./second,
            'Omega_f': 2./second,
            'U_0__star': 0.15,
            'alpha': 1.,
        })
    elif synapse == 'neutral':
        params.update({
            'Omega_d': 3./second,
            'Omega_f': 3./second,
            'U_0__star': 0.5,
            'alpha': 1.,
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

    
def Synapse_wrapper(Params):
    
    
    '''
    Syynapse wrapper: Brian2 based simulation
    
 
    
    '''

    # ----------- Load and preprocess data -----------
    # Upload the reference data that we're trying to fit.
    
    # Load data
    os.chdir(r'C:\Users\Admin\Desktop\CURRENTS')
    Ref_data = np.squeeze(pd.read_csv('I_AMPA.csv',header=None).to_numpy())
    
    
    # ---------- Initial params -----------
    start_scope()


    # simulation parameters
    simtime = 6 * second               # simulation time
    sed = 39                             # random number seed
    devices.device.seed(sed)            # set the seed for all the random number realisations
   
    
    # Tune the trace extracion window
    Pre_window = 50*ms
    Post_window = 100*ms
    
    
    
    
    
    # ---- Paramter extraction ----
    
    Y_T_t = Params[0]
    alpha_t = Params[1]
    beta_t = Params[2]
    
    # ----------------- NEURONAL EQUATIONS ----------------------
    
    
     # neuron model
    eqs_NN = Equations('''
    # dV/dt = noise + ((-gl*(V-El)-g_na*(m*m*m)*h*(V-ENa)-g_kd*(n*n*n*n)*(V-EK)+I+I_AHP)/Cm) : volt
    dV/dt = noise + (-gl*(V-El)-g_na*(m*m*m)*h*(V-ENa)-g_kd*(n*n*n*n)*(V-EK)+I-I_syn)/Cm  : volt
    dm/dt = alpha_m*(1-m)-beta_m*m : 1
    dh/dt = alpha_h*(1-h)-beta_h*h : 1
    dn/dt = (alpha_n*(1-n)-beta_n*n) : 1
    dhp/dt = 0.128*exp((17.*mV-V+VT)/(18.*mV))/ms*(1.-hp)-4./(1+exp((30.*mV-V+VT)/(5.*mV)))/ms*h : 1
    alpha_m = 0.32*(mV**-1)*4*mV/exprel((13*mV-V+VT)/(4*mV))/ms : Hz
    beta_m = 0.28*(mV**-1)*5*mV/exprel((V-VT-40*mV)/(5*mV))/ms : Hz
    alpha_h = 0.128*exp((17*mV-V+VT)/(18*mV))/ms : Hz
    beta_h = 4./(1+exp((40*mV-V+VT)/(5*mV)))/ms : Hz
    alpha_n = 0.032*(mV**-1)*5*mV/exprel((15*mV-V+VT)/(5*mV))/ms : Hz
    beta_n = .5*exp((10*mV-V+VT)/(40*mV))/ms : Hz
    noise = sigma*(2*gl/Cm)**.5*randn()/sqrt(dt) : volt/second (constant over dt)
    I : amp
    
    
    x : meter
    y : meter
    ''')
    
    
    
    
    
    # ----------------- SYNAPTIC EQUATIONS ----------------------
    
    # Basic synaptic equations are the same. what changes is how the post synaptic current behaves.
    
  
    
    eqs_Syn = Equations('''
    # Fraction of activated presynaptic receptors
    
    
    # Usage of releasable neurotransmitter per single action potential:
    du_S/dt = -Omega_f * u_S : 1 (clock-driven)
    
    # Fraction of synaptic neurotransmitter resources available for release:
    dx_S/dt = Omega_d *(1 - x_S) : 1 (clock-driven)
    dY_S/dt = -Omega_c * Y_S : mole (clock-driven)
    
    
    # Define the variables of the model
    G_A : mole  # gliotransmitter concentration in the extracellular space
    U_0 : 1
    r_S : 1     # Because r_S is the product of u_S and x_S that are event-driven, it is itself event-driven too
    
    
     
    # Astrocyte ID for connection
    astro_index : integer
    # Per-synapse gliotransmitter-effect parameter
    alpha  : 1
    ''')
    
    # -------------- Event based update --------------
    
    
    pre = '''
    
    U_0 =  U_0__star
    u_S += U_0 * (1 - u_S)
    r_S = u_S * x_S # released synaptic neurotransmitter resources
    x_S -= r_S
    Y_S += rho * Y_T * r_S
    '''
    post = None
    

    
    # ----------- UPGRADED MODEL -------------
    
    
    eqs_Syn += Equations('''
                         
                         
                            dr_ampa/dt = alpha_ampa_new * Y_S * (1 - r_ampa) - beta_ampa_new * r_ampa: 1 (clock-driven)
                            dr_nmda/dt =  alpha_nmda_new * Y_S * (1 - r_nmda) - beta_nmda_new * r_nmda : 1 (clock-driven)
                            
              
                            
                            r_ampa_tot_post = r_ampa : 1 (summed)
                            r_nmda_tot_post = r_nmda : 1 (summed)
                         
                         
                        ''')
                        

    
    eqs_NN += Equations(''' 
                        I_syn =  I_ampa + I_nmda: amp
                        I_ampa = g_ampa*(V-E_ampa)*(r_ampa_tot) : amp
                        I_nmda = g_nmda*(V-E_nmda)*(r_nmda_tot)/(1+exp(-0.062*V/mV)/3.57) : amp
                        r_nmda_tot :1
                        r_ampa_tot :1
                        
                    
                        ''')
    
    
    
    
    
    # ----------- Network build -----------
    
    params_NN = get_Neuronparam(sigma = 0*mV)
    params_Syn = get_Synparam(synapse = 'neutral', Y_T = Y_T_t *mmole, alpha_ampa_new = alpha_t * 1/mole * 1/second, beta_ampa_new = beta_t * 1/second)
   

    
    
    poisson_rate = 1*Hz
    P = PoissonGroup(1, poisson_rate)
    
    neuron = NeuronGroup(1, eqs_NN, threshold='V > 0*mV', refractory=2*ms,
                     method='exponential_euler',namespace=params_NN)
    neuron.V = -39 *mV # Initialize neuron voltage
    
    
    synapse = Synapses(P, neuron, model=eqs_Syn,
                   on_pre=pre, on_post=post,namespace=params_Syn,method='exponential_euler')
    
    
    #%
    # Connect the input spikes to the neuron
    synapse.connect(i=0, j=0) # Connect the single input "neuron" to the single target neuron
    synapse.x_S = 1.0
    # --- Monitoring ---
    state_monitor = StateMonitor(neuron, ['V','I_syn','I_ampa','I_nmda'], record=0)
    # state_monitorsynapse = StateMonitor(synapse, ['Y_S','r_ampa','r_nmda'], record=0)
    spike_monitor_neuron = SpikeMonitor(neuron)
    spike_monitor_input = SpikeMonitor(P)
    
   
    # --- Run Simulation ---
    run(simtime)
     
     
    # --- Extract traces ---
    Simulated_trace = state_monitor[0].I_ampa 
    
    # --- Extract window ---
    
    # Find the peaks
    peaksRef, _ = find_peaks(abs(Ref_data))
    peaksSim, _ = find_peaks(abs(Simulated_trace))
    
    
    # Reference data
    Start_idx = np.int16(peaksRef[0] - timestep(Pre_window, defaultclock.dt))
    End_idx = np.int16(peaksRef[0] + timestep(Post_window, defaultclock.dt))
    Ref_data_ = Ref_data[np.int16(Start_idx):np.int16(End_idx)]
    
    
    Start_idx = np.int16(peaksSim[0] - timestep(Pre_window, defaultclock.dt))
    End_idx = np.int16(peaksSim[0] + timestep(Post_window, defaultclock.dt))
    Simulated_trace_ = Simulated_trace[np.int16(Start_idx):np.int16(End_idx)]
    
    
    # Compare the two traces
     
    MSE = mean_squared_error(Ref_data_,Simulated_trace_) 
    
    
    # Normalize by the observed data's range 
    # Norm_factor = np.max(Ref_data_) - np.min(Ref_data_)
    Norm_factor = np.std(Ref_data_)
  
    # ---------- Retrun the  NMRE ----------
         
        
    return MSE / Norm_factor


def get_newspace(res_gp,pers):
    
    '''
    This function returns the narrowed search space for the Bayesian Optimization 
    algorithm given the 'pers' persentage of best points from the previous Optimization 
    Process.
  
    pers = between 0 and 1  
    
    Space parameters are set as follow:
        space = [Y_T,
                Alpha,
                Beta
                     ]
    
    
    '''
    # --- Best points ---
    n_bp = np.int16(len(res_gp.func_vals[:]) * pers)
    
    
    
    # --- Objective Function Evaluations ---
    # Draw out and sort them.
    
    # np.argsort() does not sort the array itself. Instead, it returns an array of integer indices that, 
    # if used to index the original array, would produce a sorted version of the array.
    Sorting_idx = np.argsort(res_gp.func_vals[:])
    
    # Sort function values
    Sorted_funcval = res_gp.func_vals[Sorting_idx]
    
    
    # --- Extract space values ---
    # Define storing array sizes
    n_params = np.int16(len(res_gp.x_iters[0]))
    
    
    Space_temp = np.zeros((n_bp,n_params))
    
    i = 0
    for sort_idx in Sorting_idx[0:n_bp]:
        
        
        Space_temp[i,:] = res_gp.x_iters[sort_idx]
        i = i+1
        
        
        
    
    # ------ Redefine the searching space ------
    # Find extremities
    lower_bounds = np.min(Space_temp,axis=0)
    upper_bounds = np.max(Space_temp,axis=0)
    
    
    # Define the new space
    
    # Set HPs' range
    Y_T = Real(lower_bounds[0],upper_bounds[0],name='Vescicle [Glu]')
    Alpha = Real(lower_bounds[1],upper_bounds[1],name = 'Alpha')
    Beta  = Real(lower_bounds[2],upper_bounds[2],'log-uniform', name = 'Beta')
    
    




    space = [Y_T,
             Alpha,
             Beta
        
        ]
    
    return space











from skopt import gp_minimize

# --------------------- FIRST RANDOM SEARCH --------------------- 

# # Set the parameters' range
# Y_T = Integer(200,700,name = 'Vescicle [Glu]')
# alpha = Real(1e5,1e10, name = 'Alpha')
# beta = Real(1e-4,1e3,'log-uniform', name = 'Beta')

# Set the parameters' range
Y_T = Integer(300,301,name = 'Vescicle [Glu]')
alpha = Real(1e5,1.001e5, name = 'Alpha')
beta = Real(800,801,'log-uniform', name = 'Beta')


space = [Y_T,
         alpha,
         beta,
             ]




# ------- Run the gp -------

res_gp = gp_minimize(Synapse_wrapper, space, n_calls=120, random_state=0,verbose=True)

  
# ------- Plot the Partial Dependence Plots (PDP) -------

# This is needed to spotlight possible parameters that do not have great influence
# on the model performance. These appear to have flat Partial Dependence profiles.
 #%%   
    
from skopt.plots import plot_objective
    
plt.figure()
plot_objective(res_gp,n_points=20)
plt.show()    




# ------- Narrow the search space -------

# The search space is narrowed in base of the range covered by the first 10% of best points.
# The space covered by Beta parameter is left untouched and optimized later on.
pers = 10/100
Narrowed_space = get_newspace(res_gp,pers)



#%% 
# --------------------- SECOND RANDOM SEARCH ---------------------     
res_gp2 = gp_minimize(Synapse_wrapper, Narrowed_space, n_calls=120, random_state=0,verbose=True)    


# --- PLOT ---
    
plt.figure()
plot_objective(res_gp2,n_points=20)
plt.show()  



# ----------- Step 5 ------------
# Choose median or best HP values but the ridge




def HP_choice(res_gp2,method,n_hp,xbest):
    
    '''
    The function chooses the best set of n_hp parameters given the method opted.
    The method used should depend on the type of distribution arises in the parameter space
    from the x% best values.
    
    method: 'median', 'best','automatic'
    
    xbest: persentage (0-1) of best points in the HP space used to declare the best values. 
    
    n_hp: number of hyperparameters to select. It follows the HP order firmly set 
        throughout all the code. 
        Space parameters are set as follow:
            space = [SR,
                     LR,
                     W,
                     Win,
                     Beta
                         ]
         
    
    
    '''

    # --- Best points ---
    n_bp = np.int16(len(res_gp.func_vals[:]) * pers)
    
    
    
    # --- Objective Function Evaluations ---
    # Draw out and sort them.
    
    # np.argsort() does not sort the array itself. Instead, it returns an array of integer indices that, 
    # if used to index the original array, would produce a sorted version of the array.
    Sorting_idx = np.argsort(res_gp.func_vals[:])
    
    # Sort function values
    Sorted_funcval = res_gp.func_vals[Sorting_idx]
    #%%
    
    

Sim_timeseries,reference  = Synapse_simulation(Params)
