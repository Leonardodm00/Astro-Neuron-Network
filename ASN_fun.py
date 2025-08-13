



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

from sklearn.neighbors import KDTree




def Distance_based_connections(N,params_Syn,sed):
    
    '''
    The ADJ matrix must comply to the following concention:
        rows    : pre
        columns : post
    
    '''
    
    import random
    random.seed(sed)
    
    
    # Retrieve number of neurons and pre initialize the adj matrix
    Nn = N.N
    
    ADJ = np.zeros((Nn,Nn))
    
    for neu_pre in range(Nn):
        for neu_post in range(Nn):
            
            # Sort out same neuron
            if neu_pre == neu_post:
                continue
            # Define the distance
            else:
            
            
                d = np.sqrt( (N[neu_post].x/um - N[neu_pre].x/um)**2 + (N[neu_post].y/um - N[neu_pre].y/um)**2 )
                
                # Distance dependent probability
                prob = -d * params_Syn['slope'] + params_Syn['intercept']
                
                # Generate a random number; if less than prob a connection is established
                rnd_num = random.random()
                if  rnd_num <= prob:
                    
                    ADJ[neu_pre,neu_post] = 1
                
        
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
        'conn_dist' : 150, # [um] 
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
    
    

def get_Neuronparam(Adaptation=True,delta = 0,**kwargs):
    
    
    
    Neuron_area =  300*umetre**2
    
    if Adaptation == True:
            
            g_AHP= (0.001*msiemens*cm**-2) * Neuron_area
            
            
    else:
            g_AHP = (0*msiemens*cm**-2) * Neuron_area

    
    
    
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
    'sigma': 4.1 * mV,                     # standard deviation of the noisy voltage fluctuations
    'Tau_max': 608 * ms,                # Decay factor of AHP
    
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






def get_Synparam(synapse_type='depressing',**kwargs):
    
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
        
       # Neurotransmitter release time constants
       'tau_rise_NT': 1*ms,
       'tau_decay_NT': 25*ms, # 
        
       
    
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
        
        # Params of the kinetic model post-syn
        'alpha_ampa_kin' : 1 * 1/mmole * 1/ms,
        'alpha_nmda_kin' : 0.013 * 1/mmole * 1/ms,
        'beta_ampa_kin'  : 2 * 1/ms,
        'beta_nmda_kin'  :  0.36 * 1/ms,
        'epsilon': 1e-40 * Hz,
        
        # Params of the kinetic model post-syn
        'tau_rise_ampa': 1*ms,
        'tau_decay_ampa': 2*ms,
        'tau_rise_nmda': 2*ms,
        'tau_decay_nmda': 100*ms,
        
        # Synaptic efficacy
        'Xi': 0.8,
        
        # Connection probability
        'conn_prob' : 0.107, # Random 
        
        # Distance dependent
        'slope': 1/500, # in [um] sHOULD BE 500
        'intercept': 1,
        
        
       
    
       # Asynchronous Release parameters (uncommented and added to dictionary)
       'AsynchronousRelease': False,
       'tau_ar': 700 * ms,
       'Uar': 0.003,
       'Umax': 0.5/ms,
       'x0': 5, # x0 seems to be unitless here
    
    }
    
    # Define the norm factor for the double exponential decay used for the 
    # neurotransmitter release.
    
    Norm_NT = Get_norm(params['tau_rise_NT'],params['tau_decay_NT'])
    params.update({'Norm_NT':Norm_NT})
    

    # ------------------ SYNAPSES ------------------
    if synapse_type == 'depressing':
        params.update({
            'Omega_d': 2./second,
            'Omega_f': 3.33/second,
            'U_0__star': 0.6,
            'alpha': 0.,
        })
    elif synapse_type == 'facilitating':
        params.update({
            'Omega_d': 2./second,
            'Omega_f': 2./second,
            'U_0__star': 0.15,
            'alpha': 1.,
        })
    elif synapse_type == 'neutral':
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





def Neuronal_Network(Nn,Connection_var, RandomKinetics = False, OnlyExc= True ,
                     Syn_Currents_model = 'Kinetic',add_delay= False,delay_mode= 'random',
                     Max_delay = 10*ms,ics = True,std_pers = 0.01, Simulated_network = 'Neuronal',
                     Decay_type = 'Double_exp',synapse_type = 'neutral'):
    
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
    I_cell = -gl*(V-El)-g_na*(m*m*m)*h*(V-ENa)-g_kd*(n*n*n*n)*(V-EK)-g_AHP*p*(V-EK) : amp
    # I_syn : amp
    # I_AHP = g_AHP*p*(V-EK) : volt * siemens 
    x : meter
    y : meter
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
        
        eqs_Syn = Equations('''
        
            # Usage of releasable neurotransmitter per single action potential:
            du_S/dt = -Omega_f * u_S : 1 (clock-driven)
            
            # Fraction of synaptic neurotransmitter resources available for release:
            dx_S/dt = Omega_d *(1 - x_S) : 1 (clock-driven)
            
            
            
            # Define the variables of the model
            G_A : mole  # gliotransmitter concentration in the extracellular space
            U_0 : 1
            r_S : 1     # Because r_S is the product of u_S and x_S that are event-driven, it is itself event-driven too
            
            
         
            # Astrocyte ID for connection
            astro_index : integer
            # Per-synapse gliotransmitter-effect parameter
            # alpha  : 1
            ''')
        
        # -------------- Event based update --------------
        
        
        pre = '''
        
        U_0 =  U_0__star
        u_S += U_0 * (1 - u_S)
        r_S = u_S * x_S # released synaptic neurotransmitter resources
        x_S -= r_S
        
        '''
        post = None
        
        
        
        
        
    else:
        
        eqs_Syn = Equations('''
            # Fraction of activated presynaptic receptors
            dGamma_S/dt = O_G * G_A * (1 - Gamma_S) - Omega_G * Gamma_S : 1 (clock-driven)
    
            
            # Usage of releasable neurotransmitter per single action potential:
            du_S/dt = -Omega_f * u_S : 1 (clock-driven)
            
            # Fraction of synaptic neurotransmitter resources available for release:
            dx_S/dt = Omega_d *(1 - x_S) : 1 (clock-driven)
           
            
            
            # Define the variables of the model
            G_A : mole  # gliotransmitter concentration in the extracellular space
            U_0 : 1
            r_S : 1     # Because r_S is the product of u_S and x_S that are event-driven, it is itself event-driven too
            
            
         
            # Astrocyte ID for connection
            astro_index : integer
            # Per-synapse gliotransmitter-effect parameter
            # alpha  : 1
            ''')
        
        # -------------- Event based update --------------
        
        pre = '''
        
        U_0 =  (1 - Gamma_S) * U_0__star + alpha * Gamma_S
       
        u_S += U_0 * (1 - u_S)
        r_S = u_S * x_S # released synaptic neurotransmitter resources
        x_S -= r_S
        
        '''
        post = None
        
        
        
        
        
        
    # ---------- Syn glutamate model ----------
    
    if Decay_type == 'Single_exp':
        
        
        eqs_Syn += Equations('''
                             
                             dY_S/dt = -Omega_c * Y_S : mole (clock-driven)
                             
                             ''')
        
        pre +=  '''
        
                Y_S += rho * Y_T * r_S
        
                ''' 

    if Decay_type == 'Double_exp':
        
        
        eqs_Syn += Equations('''
                             
                             
                             dY_S/dt = ((tau_decay_NT / tau_rise_NT) ** (tau_rise_NT / (tau_decay_NT - tau_rise_NT))*x_Y_S-Y_S)/tau_rise_NT : mole (clock-driven)
                             dx_Y_S/dt = -x_Y_S/tau_decay_NT                                                 : mole (clock-driven)
                             
                            
                             
                             ''')
        
        
        pre +=  '''
        
                x_Y_S += rho * Y_T * r_S
        
                ''' 

    
    # ----------- SYNAPTIC CURRENTS MODEL -------------
    
   
    if Syn_Currents_model == 'Nina':
        
        
    
            
        eqs_Syn += Equations('''
                            
                        
                        
                      
                        
                        ds_ampa/dt = -s_ampa/tau_ampa : 1 (clock-driven)
                        
                        
                        s_ampa_tot_post = s_ampa :1 (summed)
                        s_nmda_tot_post = w * S * x_d * s_nmda  :1 (summed)
                        ds_nmda/dt = -s_nmda/(taus_nmda)+alpha_nmda*x_nmda*(1-s_nmda) : 1 (clock-driven)
                        dx_nmda/dt = -x_nmda/(taux_nmda) :1 (clock-driven)
                        dx_d/dt = (1-x_d)/tau_d :1 (clock-driven)
                        
                        
                        
        
                          
                          
                         
                          ''')
                          
                          
        pre += '''           
                x_nmda += 1
                x_d *= (1-U)
                s_ampa += w * S * x_d 
                       '''                 
                          
                          
        eqs_NN += Equations(''' 
                            I_syn =  I_ampa + I_nmda: amp
                            I_ampa = g_ampa*(V-E_ampa)*(s_ampa_tot) : amp
                            I_nmda = g_nmda*(V-E_nmda)*(s_nmda_tot)/(1+exp(-0.062*V/mV)/3.57) : amp
                            s_nmda_tot :1
                            s_ampa_tot :1
                            
                        
                            ''')
    
    
    
    elif Syn_Currents_model == 'TM-coupled':
        
        eqs_Syn += Equations('''
                             
                                   
                                dr_ampa/dt = -r_ampa/tau_decay_ampa : 1 (clock-driven)
                                
                                dr_nmda/dt = ((tau_decay_nmda / tau_rise_nmda) ** (tau_rise_nmda / (tau_decay_nmda - tau_rise_nmda))*x_r_nmda-r_nmda)/tau_rise_nmda : 1 (clock-driven)
                                dx_r_nmda/dt = -x_r_nmda/tau_decay_nmda   : 1 (clock-driven)
                                
                               
                                r_ampa_tot_post = r_ampa : 1 (summed)
                                r_nmda_tot_post = r_nmda : 1 (summed)
                             
                             
                            ''')
                            
                            
        pre += '''           
                r_ampa +=  (alpha_ampa_kin * rho * Y_T * r_S * Xi)/(alpha_ampa_kin * rho * Y_T * r_S * Xi + beta_ampa_kin) 
                x_r_nmda +=  (alpha_nmda_kin * rho * Y_T * r_S * Xi)/(alpha_nmda_kin * rho * Y_T * r_S * Xi + beta_nmda_kin) 
               
                        '''          

        
        
        eqs_NN += Equations(''' 
                            I_syn =  I_ampa + I_nmda: amp
                            I_ampa = g_ampa*(V-E_ampa)*(r_ampa_tot) : amp
                            I_nmda = g_nmda*(V-E_nmda)*(r_nmda_tot)/(1+exp(-0.062*V/mV)/3.57) : amp
                            r_nmda_tot :1
                            r_ampa_tot :1
                            
                        
                            ''')

    
    else:
        
        eqs_Syn += Equations('''
                  
                             
                                   
                                dr_ampa/dt = alpha_ampa_kin * Y_S * (1 - r_ampa) - beta_ampa_kin * r_ampa: 1 (clock-driven)
                                dr_nmda/dt =  alpha_nmda_kin * Y_S * (1 - r_nmda) - beta_nmda_kin * r_nmda : 1 (clock-driven)
                                
                     
                                
                                
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
    
    
    
    
    # ----------- SYNAPTIC PARAMETERS ------------
    params_Syn = get_Synparam(synapse_type=synapse_type)
    
    
    
    # -------------- Currents --------------
    
    if RandomKinetics == True:
        # Some variance is given to the time constant of synaptic transmission
        
        # Determine the number of synapses
        N_syn = len(source)
        
        if OnlyExc == True:
            
            
            
            # ---------- NINA'S MODEL ----------
    
            
            x_NMDA = np.random.normal(loc=params_Syn['taux_nmda'], scale=params_Syn['taux_nmda']*std_pers, size=N_syn)
            AMPA_decay = np.random.normal(loc= params_Syn['tau_ampa'], scale= params_Syn['tau_ampa']*std_pers, size=N_syn)
            s_NMDA = np.random.normal(loc=params_Syn['taus_nmda'], scale=params_Syn['taus_nmda']*std_pers, size=N_syn)
            
            
            # Generate random weigths
            
                # Step 1-3: Generate and shift/scale the random number
            random_value = 1.0 + params_Syn['sd'] * np.random.randn()
        
            # Step 4: Clip the value
            w_ = np.clip(random_value, min_val, max_val)
            
            
            
            
            
            
          
            params_Syn.update({  
                
                
                'taux_nmda': x_NMDA,
                'tau_ampa': AMPA_decay,
                'taus_nmda': s_NMDA,
                'w':w_
                
                
                })
            
            
            # ---------- KINETIC MODEL ----------
            
            
            alpha_nmda_ = np.random.normal(loc=params_Syn['alpha_nmda_kin'], scale=params_Syn['alpha_nmda_kin']*std_pers, size=N_syn)
            alpha_ampa_ = np.random.normal(loc= params_Syn['alpha_ampa_kin'], scale= params_Syn['alpha_ampa_kin']*std_pers, size=N_syn)
            
            beta_nmda_ = np.random.normal(loc=params_Syn['beta_nmda_kin'], scale=params_Syn['beta_nmda_kin']*std_pers, size=N_syn)
            beta_ampa_ = np.random.normal(loc= params_Syn['beta_ampa_kin'], scale= params_Syn['beta_ampa_kin']*std_pers, size=N_syn)
            
                    
    
        
            params_Syn.update({
                
                'alpha_ampa_kin' :  alpha_ampa_ * 1/mmole * 1/ms,
                'alpha_nmda_kin' :  alpha_nmda_ * 1/mmole * 1/ms,
                'beta_ampa_kin'  :   beta_ampa_ * 1/ms,
                'beta_nmda_kin'  :   beta_nmda_ * 1/ms
                
                })
    
    

    
    # -------------------------- INITIALIZE THE NETWORKS ---------------------------
    
    # ---- Get parameters ----
    params_NN = get_Neuronparam(Adaptation)
    
    
    N = NeuronGroup(Nn, model=eqs_NN, name='Neuron*',namespace= params_NN, threshold='V>20*mV', refractory=2 * ms,
                        method='exponential_euler')
    
    # Initialize neuron parameters
    N.V = -39 * mV                          # approximately resting membrane potential
    N.I = '(rand() -0.5) * I_inj'          # Make neurons heterogeneously excitable
    
    
    
    # ----- SET POSITIONS AND CONNECTIONS -----
    # Position neurons on a grid

    Coordinates = get2D_rnd_coordinates(N.N,c_min,c_max,sed)
    N.x = Coordinates[:,0]*um
    N.y = Coordinates[:,1]*um
    
    
    
    
    S = Synapses(N, model=eqs_Syn,
                        on_pre=pre,
                        on_post=post,
                        name='Synapse*',
                        namespace=params_Syn,
                        method='exponential_euler')
    
    
    
    # -------------- Connections --------------
    # Check the type of connection scheme and implement the connection
    if isinstance(Connection_var, str):
        
        
        if Connection_var == 'Random':
            # --- Random ---
            S.connect(p=params_Syn['connprob'])
            
        elif Connection_var == 'Distance':
            
            ADJ_ = Distance_based_connections(N,params_Syn,sed)
            
                
            Source_neuron,Target_neuron = unfold_ADJ(ADJ_)
            
            S.connect(i=Source_neuron, j=Target_neuron)
        
      
          
        
        
        
        
        
    elif isinstance(Connection_var, numpy.ndarray):
        
    
    
        Source_neuron,Target_neuron = unfold_ADJ(Connection_var)
        
        S.connect(i=Source_neuron, j=Target_neuron)
    
    
    
    
    
    
    # --- Delays ---
    if add_delay == True:
        
        Max_delay = Max_delay # Max conduction delay expressed in ms
        N_synapses = S.N
        
        if delay_mode == 'random':
            # Randomly generate scaling values.
            random_values = np.random.rand(N_synapses)
            Delays = (Max_delay * ms) * random_values
            S.delay = Delays
            
        elif delay_mode == 'distance':    
            # Retrieve the distances between neuronal somata.
            # Unmyelinated nerve fibers. Usually reach 0.5 [mm/ms] speed.
            # To this it must be add the time required from the vescicle release 
            # at the pre-synaptic site.
            random_values = np.random.rand(N_synapses)
            Delays = (Max_delay * ms) * random_values
            Conductance_velocity = 0.5 * (mm/ms) + Delays
            S.delay = '(sqrt((x_pre - x_post)**2 + (y_pre - y_post)**2 + (z_pre - z_post)**2)) * Conductance_velocity'
    
    
    
    
        
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
    x_pos = np.array(Astrocyte_group[:].x/um)
    y_pos = np.array(Astrocyte_group[:].y/um)
    
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
    
    
    
    
    

def Astrocyte_Group(N_astro,Connection_var,Simulated_network,sed):
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
        
        
    # ----- SET POSITIONS -----
    # Position neurons on a grid
    Coordinates = get2D_rnd_coordinates(N_astro,Params_astroGT['c_min'],Params_astroGT['c_max'],sed)
    Astro.x = Coordinates[:,0]*um
    Astro.y = Coordinates[:,1]*um
        
        
    # ----- Gap-junction based astro links -----
    
    Gap_Eq = Equations('''
                     
            
                        delta_I = I_post - I_pre: mole
                        I_coupling = -F/2*(1 + tanh((abs(delta_I) - I_Theta)/omega_I))*sign(delta_I) : mole/second 
                        I_coupling_tot_post = I_coupling : mole/second (summed)
                        
                       
                        ''')
    
    GJ = Synapses(Astro,Astro,
                  model=Gap_Eq,
                  method='exponential_euler',
                  namespace= Params_astroGT,
                  name = 'Gap_junctions*'
                  )
    
    
    
    
    # ----- Connections -----
    '''
    Astros are connected by gap-junctions within distance of 100 um
    
    Paper: A Computational Model of Interactions Between Neuronal and 
        Astrocytic Networks: The Role of Astrocytes in the Stability of the Neuronal Firing Rate
    
    '''
    
    if isinstance(Connection_var, str):
    
    
        # --------- RANDOM -----------
        Source,Target = astrocyte_connections(Astro,Params_astroGT['conn_dist'])
        GJ.connect(i=Source , j= Target)
    
    elif isinstance(Connection_var, numpy.ndarray):
        # -------- MATRIX BASED --------
        Source,Target = unfold_ADJ(Connection_var)
        
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
def Gliotransmission(N_astro,ics,Astro):
    
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
                            method='rk4',
                            name='gliot_release*',
                            namespace=Params_astroGT)
    
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

def Synapse_to_astro(synapse,Astro,ADJ):
    # ---- Syn-astro ----
    
    Syn_Astro = Synapses(synapse,Astro,
                        model='''
                        # neurotransmitter concentration in the extracellular space
                        Y_extra_post = Y_S_pre : mole (summed)
                        ''',
             
                        
                        name="ecs_syn_to_astro*")
    
    # ---- Connections ----
    
    Source_syn,Target_astro = unfold_ADJ(ADJ)
    
    Syn_Astro.connect(i = Source_syn, j = Target_astro)

    return Syn_Astro



def Astro_to_Syn(Glio_release,synapse,ADJ):
    # ---- Astro-syn ----
    # Glio_relaease is the reference neuronal group
    Astro_Syn = Synapses(Glio_release,synapse,
                             model='''
                             # gliotransmitter concentration in the extracellular space
                             G_A_post = G_A_pre : mole (summed)
                             ''',
                  
                             name="ecs_astro_to_syn*"
                             )
    
    # ---- Connections ----
    
    Source_astro,Target_syn = unfold_ADJ(ADJ)
    
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
                
                
                # alpha_,beta_ = TF_params(NN_dist[neu])
                
                
                alpha_ = 1
                beta_ = 1
                
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
    
    APs_time = []
    APs_unit = []
    # voltagetraces = zeros((len(Traces),len(Traces[0])))
    Raster = zeros((len(Traces),len(Traces[0])))
    
    for k in range(len(Traces)):
        Trace_temp = Traces[k]
        # Subtract the mean
        Trace_temp = Trace_temp - np.mean(Trace_temp)
        Voltagefilt = signal.filtfilt(b, a, Trace_temp)  # high pass filter
        threshold = 4 * np.std(Voltagefilt)      #threshold to detect APs
        APstemp, _ = find_peaks(abs(Voltagefilt), height=threshold)
        for j in range(len(APstemp)):
            APs_time = np.append(APs_time, APstemp[j])
            APs_unit = np.append(APs_unit,k)
        # voltagetraces[k, :] = Voltagefilt

       
        
        Raster[k,APstemp] = 1
        
        
        
        
    if Visible:
        
        
            # Create the plot
        plt.figure()
        
        # Plot the unit indices (y-axis) against the spike times (x-axis)
        plt.scatter(APs_time/fs, APs_unit, s=5, marker='|')
        
        # Customize the plot
        plt.title('Spiking Activity (Raster Plot)')
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
    
    
def Electrode_traces(pitch,pitch_recsites,shift,N,MonitorN,electrode_dist,neuron_radius,electrode_radius):
    
    '''
    MEA_dict = dictionary with electrodes info: position and recordin sites
    Traces = List of Lists that contain the recordings.
    
    
    '''
    
    
    
    Grid = Get_12grid(pitch)
    
    MEA_dict = Recording_sites(pitch_recsites,shift,Grid)
    
    # --- Plot Device + Neurons
    
    
    Plot_CultureDevice(Grid,N,Nn)
    
    Traces = Electrode_recording(MEA_dict,N,MonitorN,electrode_dist,neuron_radius,electrode_radius)
    
    
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







