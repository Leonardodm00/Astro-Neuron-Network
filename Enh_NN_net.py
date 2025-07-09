
"""
Created on Wed Jul  9 14:47:32 2025

@author: Admin
"""

# -*- coding: utf-8 -*-
"""
Created on Wed Jul  9 10:28:10 2025

@author: leona
"""
import matplotlib.pyplot as plt
from brian2 import *


def get_Neuronparam(Adaptation):
    
    
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
        'rho_c': 0.005,            # synaptic vesicle-to-extracellular space volume ratio
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
    
    # Post-synaptic neuron parameters
    params.update({
        'G_e' : 2*mV,   # Max synaptic depolarization
        'tau_m': 20*ms,  # Membrane time constant
        'tau_r': 5*ms,   # Refractory time
        
      
        # --- AMPA/NMDA means ---
        'AMPA_tau_e_r': 0.5*ms, # excitatory conductance rise time
        'AMPA_tau_e': 5*ms,   # excitatory conductance time constant
        'NMDA_tau_e_r': 10*ms, # excitatory conductance rise time
        'NMDA_tau_e': 40*ms,   # excitatory conductance time constant
        
        # --- AMPA/NMDA std ---
        'AMPA_tau_e_r_STD': 0.5*ms, # excitatory conductance rise time
        'AMPA_tau_e_STD': 5*ms,   # excitatory conductance time constant
        'NMDA_tau_e_r_STD': 10*ms, # excitatory conductance rise time
        'NMDA_tau_e_STD': 40*ms,   # excitatory conductance time constant
        
        # --- GABA means ---
        'GABA_tau_i_r': 10*ms,  # inhibitory conductance time constant
        'GABA_tau_i': 10*ms,  # inhibitory conductance time constant
        
        # --- GABA std ---
        'GABA_tau_i_r_STD': 10*ms,  # inhibitory conductance time constant
        'GABA_tau_i_STD': 10*ms,  # inhibitory conductance time constant
        
        
        # Synaptic efficacy 
        
        'we': 4, # Total excitatory synaptic weigth
        'we_std' : 0.2,  # Total excitatory synaptic weight std
        'wi': - 1,
        'wi_std': 0.1,
        
        
        
        
        
        
        # 'E_L': -60*mV,   # reversal potential
        # 'V_th': -55*mV,   # Threshold
        # 'V_r': -57*mV,   # Reset
    })
    
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
        'sigma'  : 2.8284, # variance in the diffusion approx,
        'beta'   : 0.5,
        'b'      : 5.
    })
    

    
    params.update(kwargs)

    return params




# Ensure a clean Brian2 state for repeatable simulations
start_scope()





# ------------------------- SET OPTIONS -------------------------

# ------- Synapses -------
synapse='neutral'
ics=None 
dt=None            
Name_syn='Synapses*'
postc_sic='exp'
sic=None 
delay=None
RandomKinetics = True 
OnlyExc = True

Max_delay = 25 *ms
add_delay = False
delay_mode = 'random'


# ------- Neurons -------
Adaptation = True
Name_neu='Neurons*'

# ------- Astrocytes -------
oscillations = 'AM'



# ------------------------- PARAMETERS -------------------------
# SET PARAMETERS

# simulation parameters
simtime = 100 * second               # simulation time
transient = 3 * second              # time omitted as transient
sed = 39                             # random number seed
devices.device.seed(sed)            # set the seed for all the random number realisations



# --------- NEURONAL -----------
Nl = 2
N = Nl * Nl  

# area = 300*umetre**2                # membrane area of the neuron
# Cm = (2*ufarad*cm**-2) * area       # membrane capacitance
# El = -39.2 * mV                     # Nernst potential of leaky ions
# EK = -80 * mV                       # Nernst potential of potassium
# ENa = 70 * mV                       # Nernst potential of sodium
# g_na = 1.6 * 50 * msiemens * cm ** -2 * area  # maximal conductance of sodium channels
# g_kd = 1.3 * 5 * msiemens * cm ** -2 * area   # maximal conductance of potassium
# gl = (0.3*msiemens*cm**-2) * area   # maximal leak conductance
# VT = -30.4*mV                       # alters firing threshold of neurons
# sigma = 4.5 * mV                      # standard deviation of the noisy voltage fluctuations

# if adaptation == True:
        
#         g_m= (0.4*msiemens*cm**-2) * area
        
        
# else:
#         g_m = 0




# --------- SYNAPTIC -----------

# ----- Connectivity -----
source = [0,1,2]
target = [1,2,3]




# # --- Synaptic dynamics
# E_Iper = 80/100,          # Persentage of excitatory connections
# # Omega_d (see below)      # Depression rate
# # Omega_f (see below)      # Facilitation rate,
# # U_0__star (see below)    # Basal synaptic release probability
# Omega_c = 40./second,     # Neurotransmitter clearance rate
# rho_c = 0.005,            # synaptic vesicle-to-extracellular space volume ratio
# Y_T = 500.*mmole,         # Total neurotransmitter synaptic resource (in terms of vesicular concentration)
# # --- Presynaptic receptors
# O_G = 1.5/umole/second,   # Agonist binding rate (activating)
# Omega_G = 0.5/(60*second),# Agonist release rate (inactivating)
# # alpha (see below)        # Gliotransmitter effect on synaptic release
# # --- SIC/SOC
# G_sic =  4.5*mV,      # Max SIC/SOC depolarization
# tau_sic_r = 30.*ms,      # SIC/SOC rise time constant
# tau_sic = 600.*ms,       # SIC/SOC decay time constant


# if oscillations == 'AM':
    
#         K_P = 0.1*umole,
#         O_delta = 0.01*umole/second,
#         Omega_5P = 0.1/second,
#         I_bias = 0.8*umole
    
# elif oscillations == 'FM':
    
#         K_P = 0.05*umole,
#         O_delta = 0.05*umole/second,
#         Omega_5P =  0.1/second,
#         I_bias = 1.*umole
    
# else:
#     raise ValueError('oscillations argument has to be "AM" or "FM"')


# # ------------------ SYNAPSES ------------------
# if synapse == 'depressing':
    
#         Omega_d = 2./second,
#         Omega_f = 3.33/second,
#         U_0__star = 0.6,
#         alpha = 0.,
    
# elif synapse == 'facilitating':
    
#         Omega_d = 2./second,
#         Omega_f = 2./second,
#         U_0__star = 0.15,
#         alpha = 1.,
    
# elif synapse == 'neutral':
    
#         Omega_d = 3./second,
#         Omega_f = 3./second,
#         U_0__star = 0.5,
#         alpha =  1.,
    
# else:
#     raise ValueError('synapse argument has to be "depressing", "facilitating" or "neutral"')

# # Post-synaptic neuron parameters

# G_e = 2*mV,   # Max synaptic depolarization
# tau_m= 20*ms,  # Membrane time constant
# tau_r= 5*ms,   # Refractory time
  
# # --- AMPA/NMDA means ---
# AMPA_tau_e_r= 0.5*ms, # excitatory conductance rise time
# AMPA_tau_e= 5*ms,   # excitatory conductance time constant
# NMDA_tau_e_r= 10*ms, # excitatory conductance rise time
# NMDA_tau_e= 40*ms,   # excitatory conductance time constant

# # --- AMPA/NMDA std ---
# AMPA_tau_e_r_STD= 0.5*ms, # excitatory conductance rise time
# AMPA_tau_e_STD= 5*ms,   # excitatory conductance time constant
# NMDA_tau_e_r_STD= 10*ms, # excitatory conductance rise time
# NMDA_tau_e_STD= 40*ms,   # excitatory conductance time constant

# # --- GABA means ---
# GABA_tau_i_r= 10*ms,  # inhibitory conductance time constant
# GABA_tau_i= 10*ms,  # inhibitory conductance time constant

# # --- GABA std ---
# GABA_tau_i_r_STD= 10*ms,  # inhibitory conductance time constant
# GABA_tau_i_STD= 10*ms,  # inhibitory conductance time constant


# # Synaptic efficacy 
# we_AMPA = 0.5, # Relative contribution of AMPA channels to the total syn weight
# we_NMDA = 0.5, # Relative contribution of NMDA channels to the total syn weight
# we= 4, # Total excitatory synaptic weigth
# we_std = 0.2,  # Total excitatory synaptic weight std
# wi= - 1,
# wi_std= 0.1,




# -------- ASTROCYTIC ---------
# ----Input
f_in = 1.*Hz,              # Input frequency (synapse)
f_c = 1.*Hz,              # Input frequency (gliotransmission)
# 't_on' : 0*second,         # Start of synaptic stimulation (used in STDP)
t_off =  Inf*second,      # End of astrocyte stimulation (used in standalone gliotransmission)
# --- IP_3R kinectics
d_1 = 0.13*umole,         # IP_3 binding affinity
O_2 = 0.2/umole/second,   # Inactivating Ca^2+ binding rate
d_2 = 1.05*umole,         # Inactivating Ca^2+ binding affinity
d_3 = 0.9434*umole,       # IP_3 binding affinity (with Ca^2+ inactivation)
d_5 = 0.08*umole,         # Activating Ca^2+ binding affinity
# ---  Calcium fluxes
C_osc = 0.2*umole,        # Estimated Threshold for Ca^2+ oscillations
C_T = 2*umole,            # Total ER Ca^2+ content
rho_A = 0.18,             # ER-to-cytoplasm volume ratio
Omega_C = 6/second,       # Maximal Ca^2+ release rate by IP_3Rs
Omega_L =  0.1/second,     # Maximal Ca^2+ leak rate,
O_P = 0.9*umole/second,   # Maximal Ca^2+ uptake rate
# K_P (see below)          # Ca^2+ affinity of SERCA pumps
# --- IP_3 production
# Omega_delta (see below)  # Maximal rate of IP_3 production by PLCdelta
K_delta = 0.5*umole,      # Ca^2+ affinity of PLCdelta
kappa_delta = 1.*umole,    # Inhibiting IP_3 affinity of PLCdelta
# --- IP_3 degradation
# Omega_5P (see below)     # Maximal rate of IP_3 degradation by IP-5P
O_3K =  4.5*umole/second,  # Maximal rate of IP_3 degradation by IP_3-3K
K_D = 0.5*umole,          # Ca^2+ affinity of IP3-3K
K_3K = 1.*umole,           # IP_3 affinity of IP_3-3K
# --- IP_3 diffusion
F = 2.*umole/second,       # GJC IP_3 permeability (nonlinear)
I_Theta = 0.3*umole,      # Threshold IP_3 gradient for diffusion
omega_I= 0.05*umole,     # Scaling factor of diffusion
# I_bias (see below)       # IP_3 bias
# --- Agonist-dependent IP_3 production
O_beta = 1.*umole/second,  # Maximal rate of IP_3 production by PLCbeta
O_N = 0.3/umole/second,   # Agonist binding rate
Omega_N = 1.8/second,     # Inactivation rate of GPCR signalling
K_KC = 0.5*umole,         # Ca^2+ affinity of PKC
zeta =  2.,                # Maximal reduction of receptor affinity by PKC
n = 1.,                   # Cooperativity of agonist binding reaction
# --- Gliotransmitter release and time course        
C_Theta = 0.5*umole,      # Ca^2+ threshold for exocytosis
Omega_A = 0.6/second,     # Gliotransmitter recycling rate
U_A = 0.6,                # Gliotransmitter release probability
G_T = 200.*mmole,         # Total vesicular gliotransmitter
rho_e = 6.5e-4,           # Ratio of astrocytic vesicle volume/ESS volume
Omega_e = 5./second,      # Gliotransmitter clearance rate (think about distributed release)














#%
# ---------------------- NEURONAL GROUP ----------------------
 # neuron model
eqs_NN = Equations('''
# dV/dt = noise + ((-gl*(V-El)-g_na*(m*m*m)*h*(V-ENa)-g_kd*(n*n*n*n)*(V-EK)+I-I_syn+I_AHP)/Cm) : volt
dV/dt = noise + ((-gl*(V-El)-g_na*(m*m*m)*h*(V-ENa)-g_kd*(n*n*n*n)*(V-EK)+I)/Cm) : volt
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
ge_AMPA_tot : amp 
ge_NMDA_tot : amp 
I_syn =  ge_AMPA_tot * (1 - we_AMPA) + ge_NMDA_tot * (1 - we_NMDA) : amp




x : meter
y : meter
''')

# ---- Get parameters ----
params_NN = get_Neuronparam(Adaptation)


P = NeuronGroup(N, model=eqs_NN, name=Name_neu,namespace= params_NN, threshold='V>0*mV', refractory=2 * ms,
                    method='exponential_euler')

# Initialize neuron parameters
P.V = -39 * mV                          # approximately resting membrane potential
P.I = '(rand() -0.5) * 10 * pA'          # Make neurons heterogeneously excitable




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



# !!!TODO: LAter add: 
# dGamma_S/dt = O_G * G_A * (1 - Gamma_S) - Omega_G * Gamma_S : 1 (clock-driven)
# U_0 = (1 - Gamma_S) * U_0__star + alpha * Gamma_S
# ie_AMPA : amp
# ie_NMDA : amp
# B_syn_AMPA : Hz
# B_syn_NMDA : Hz
 



# nge_AMPA : 1
# nge_NMDA : 1
# AMPA_tau_e_r : second
# NMDA_tau_e_r : second
# AMPA_tau_e : second
# NMDA_tau_e : second
# we : 1
# Omega_f : 
# Omega_d :
# Omega_c :
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
Y_S += rho_c * Y_T * r_S
'''
post = None

params_Syn = get_Synparam()



# -------------- Currents --------------
# Modelled both AMPA and NMDA and GABA
# Also the weights are added

if RandomKinetics == True:
    # Some variance is given to the time constant of synaptic transmission
    
    # Determine the number of synapses
    N_syn = len(source)
    
    if OnlyExc == True:
        
        AMPA_rise = np.random.normal(loc=params_Syn['AMPA_tau_e_r'], scale=params_Syn['AMPA_tau_e_r_STD'], size=N_syn)
        NMDA_rise = np.random.normal(loc=params_Syn['NMDA_tau_e_r'], scale=params_Syn['NMDA_tau_e_r_STD'], size=N_syn)
        AMPA_decay = np.random.normal(loc=params_Syn['AMPA_tau_e'], scale=params_Syn['AMPA_tau_e_STD'], size=N_syn)
        NMDA_decay = np.random.normal(loc=params_Syn['NMDA_tau_e'], scale=params_Syn['NMDA_tau_e_STD'], size=N_syn)
        we = np.random.normal(loc=params_Syn['we'], scale=params_Syn['we_std'], size=N_syn)
        
        
        # JUST AMPA USED
        params_Syn.update({  
            
            'AMPA_tau_e_r': AMPA_rise,
            'NMDA_tau_e_r': NMDA_rise,
            'AMPA_tau_e': AMPA_decay,
            'NMDA_tau_e': NMDA_decay,
            'we':we
            
            })
                
            
            
            
            
            
        
        
        
        
        
    else:
        
        N_syn_exc = np.int16(N_syn * params_Syn['E_Iper'])
        N_syn_inh = N_syn - N_syn_exc
        
        
        AMPA_rise = np.random.normal(loc=params_Syn['AMPA_tau_e_r'], scale=params_Syn['AMPA_tau_e_r_STD'], size=N_syn_exc)
        NMDA_rise = np.random.normal(loc=params_Syn['NMDA_tau_e_r'], scale=params_Syn['NMDA_tau_e_r_STD'], size=N_syn_exc)
        AMPA_decay = np.random.normal(loc=params_Syn['AMPA_tau_e'], scale=params_Syn['AMPA_tau_e_STD'], size=N_syn_exc)
        NMDA_decay = np.random.normal(loc=params_Syn['NMDA_tau_e'], scale=params_Syn['NMDA_tau_e_STD'], size=N_syn_exc)
        
        GABA_rise = np.random.normal(loc=params_Syn['GABA_tau_i_r'], scale=params_Syn['GABA_tau_i_r_STD'], size=N_syn_inh)
        GABA_decay = np.random.normal(loc=params_Syn['GABA_tau_i'], scale=params_Syn['GABA_tau_i_STD'], size=N_syn_inh)
        
        we = np.random.normal(loc=params_Syn['we'], scale=params_Syn['we_std'], size=N_syn_exc)
        wi = np.random.normal(loc=params_Syn['we'], scale=params_Syn['we_std'], size=N_syn_inh)
        
        
        ###### TO FINISH
    
    
        



'''
# ADD THE NMDA AND AMPA and GABA

Given that the rise and decay time constant can be differntly assigned to each neuron,
Brian2 requires to initialize them as state variables 
'''

# Add case-specific code
if postc_sic =='exp' :
    eqs_Syn += Equations('''
                        
                      dge_AMPA/dt = -ge_AMPA/AMPA_tau_e + Y_S* we/AMPA_tau_e : 1
                      ge_AMPA_tot_post = ge_AMPA : amp (summed) 
                    
                      ''')
elif postc_sic =='double-exp':
    nge_AMPA = peak_normalize(1.,AMPA_tau_e_r,AMPA_tau_e)
    nge_NMDA = peak_normalize(1.,NMDA_tau_e_r,NMDA_tau_e)
    eqs_Syn += Equations('''
                    dge_AMPA/dt = -ge_AMPA/AMPA_tau_e_r + nge_AMPA*B_syn_AMPA : 1 
                    dB_syn_AMPA/dt = -B_syn_AMPA/AMPA_tau_e + Y_S* we : Hz  # Technically you need the concentration in the cleft, but you can use Y_S and rescale w/in 'w'
                    ge_AMPA_tot_post = ge_AMPA : amp (summed) 
                   
                    dge_NMDA/dt = -ge_NMDA/NMDA_tau_e_r + nge_NMDA*B_syn_NMDA : 1 
                    dB_syn_NMDA/dt = -B_syn_NMDA/NMDA_tau_e + Y_S*we : Hz  # Technically you need the concentration in the cleft, but you can use Y_S and rescale w/in 'w'
                    ge_NMDA_tot_post = ge_NMDA : amp (summed) 
                    
                    ''')


# if sic==True:
#     if postc_sic =='exp':
#         eqs_Syn += Equations('''
#                         dgsic/dt = -gsic/tau_sic + G_A_sic*wa/tau_sic : 1
#                         G_A_sic : mole
#                         ''')
#     elif postc_sic =='double-exp':
#         nsic = peak_normalize(1.,tau_sic_r,tau_sic)
#         eqs_Syn += Equations('''
#                         dgsic/dt = -gsic/tau_sic_r + nsic*B_sic : 1
#                         dB_sic/dt = -B_sic/tau_sic + G_A_sic*wa : Hz  # Technically you need the concentration in the cleft, but you can use Y_S and recale w/in 'we'
#                         G_A_sic : mole
#                         ''')



    
    



'''

ADD THE NMDA AND AMPA and GABA



'''







#!!! TO CHECK
S = Synapses(P, model=eqs_Syn,
                    on_pre=pre,
                    on_post=post,
                    name=Name_syn,
                    namespace=params_Syn,
                    dt=dt,method='exponential_euler')



# -------------- Connections --------------
S.connect(i=source, j=target)

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
    
    
    
    
    
    
    
    
    
# ----------- CREATE NETWORK OBJECT  
    
# NN_net = Network(P,S)

# --- Prepare the network ---

# NN_net.before_run()

    



#%

# --- Run try ---
# Define the state monitor
recording_stringP = ['V']
recording_stringS = ['u_S','x_S','Y_S']
dt2 = defaultclock.dt                                           # Allows for chanching the timestep of recording
synapses
traceS = StateMonitor(S, recording_stringS, record=True, dt=dt2)
trace = StateMonitor(P, recording_stringP, record=True, dt=dt2)
spikes = SpikeMonitor(P)

# NN_net.run(simtime, report='text', profile=True)
run(simtime, report='text', profile=True)



# PLOTS

#%%
fig, (ax1, ax2, ax3,ax4) = plt.subplots(4, 1,figsize=[10,12]) # Added figsize for better viewing



ax1.plot(trace.t / second, trace[0].V / mV, 'k', linewidth=0.7)


ax2.plot(trace.t / second, trace[1].V / mV, 'k', linewidth=0.7)




ax3.plot(trace.t / second, trace[2].V / mV, 'k', linewidth=0.7)




ax4.plot(trace.t / second, trace[3].V / mV, 'k', linewidth=0.7)

fig.show()

plt.figure(dpi=200)
plt.plot(spikes.t / second, spikes.i, '.k', ms=0.7)

show()

#%%



fig, (ax1, ax2, ax3) = plt.subplots(3, 1,figsize=[10,12]) # Added figsize for better viewing



ax1.plot(traceS.t / second, traceS[0].u_S / mV, 'r', linewidth=0.7)

ax1.plot(traceS.t / second, traceS[0].x_S / mV, 'c', linewidth=0.7)

# ax1.plot(traceS.t / second, traceS[0].Y_S / mV, 'k', linewidth=0.7)




ax2.plot(traceS.t / second, traceS[1].u_S / mV, 'r', linewidth=0.7)

ax2.plot(traceS.t / second, traceS[1].x_S / mV, 'c', linewidth=0.7)

# ax2.plot(traceS.t / second, traceS[1].Y_S / mV, 'k', linewidth=0.7)



ax3.plot(traceS.t / second, traceS[2].u_S / mV, 'r', linewidth=0.7)

ax3.plot(traceS.t / second, traceS[2].x_S / mV, 'c', linewidth=0.7)

# ax3.plot(traceS.t / second, traceS[2].Y_S / mV, 'k', linewidth=0.7)

# fig.show()

# plt.figure(dpi=200)
# plt.plot(spikes.t / second, spikes.i, '.k', ms=0.7)

# show()











