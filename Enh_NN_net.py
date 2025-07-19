

# -*- coding: utf-8 -*-
"""
Created on Wed Jul  9 10:28:10 2025

@author: leona
"""
import matplotlib.pyplot as plt
from brian2 import *






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





# Ensure a clean Brian2 state for repeatable simulations
start_scope()





# ------------------------- SET OPTIONS -------------------------

# ------- Synapses -------
synapse='neutral'
ics=None 
dt=None            
Name_syn='Synapses*'
postc_sic='double-exp'
sic=None 
delay=None
RandomKinetics = False 
OnlyExc = True
Syn_Currents_model = 'Nina'
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




# --------- SYNAPTIC -----------

# ----- Connectivity -----
source = [0,1,2]
target = [1,2,3]



#%
# ---------------------- NEURONAL GROUP ----------------------
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
peak_normalize = lambda peak, taur, taud : peak*(1./taud - 1./taur)/Hz/( (taur/taud)**(taud/(taud-taur))-(taur/taud)**(taur/(taud-taur)) )
if RandomKinetics == True:
    # Some variance is given to the time constant of synaptic transmission
    
    # Determine the number of synapses
    N_syn = len(source)
    
    if OnlyExc == True:

        
        x_NMDA = np.random.normal(loc=params_Syn['taux_nmda'], scale=params_Syn['taux_nmda_std'], size=N_syn)
        AMPA_decay = np.random.normal(loc=params_Syn['tau_ampa'], scale=params_Syn['tau_ampa_std'], size=N_syn)
        s_NMDA = np.random.normal(loc=params_Syn['taus_nmda'], scale=params_Syn['taus_nmda_std'], size=N_syn)
        
        
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


## NINA POST_SYNAPTIC MODEL
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



# ----------- UPGRADED MODEL -------------

else:
    
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




'''

ADD THE NMDA AND AMPA and GABA



'''

# -------------------------- INITIALIZE THE NETWORKS ---------------------------

# ---- Get parameters ----
params_NN = get_Neuronparam(Adaptation)


P = NeuronGroup(N, model=eqs_NN, name=Name_neu,namespace= params_NN, threshold='V>0*mV', refractory=2 * ms,
                    method='exponential_euler')

# Initialize neuron parameters
P.V = -39 * mV                          # approximately resting membrane potential
P.I = '(rand() -0.5) * 10 * pA'          # Make neurons heterogeneously excitable







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
recording_stringS = ['u_S','x_S','Y_S','I_syn','I_ampa','I_nmda']




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



ax1.plot(traceS.t / second, traceS[0].u_S, 'r', linewidth=0.7,label='Ready-to-relase resources')

ax1.plot(traceS.t / second, traceS[0].x_S, 'c', linewidth=0.7,label='Available resources')


ax1.legend()



ax2.plot(traceS.t / second, traceS[1].u_S, 'r', linewidth=0.7)

ax2.plot(traceS.t / second, traceS[1].x_S, 'c', linewidth=0.7)




ax3.plot(traceS.t / second, traceS[2].u_S, 'r', linewidth=0.7)

ax3.plot(traceS.t / second, traceS[2].x_S, 'c', linewidth=0.7)

ax3.set_xlabel('Time [s]')
# fig.show()

# plt.figure(dpi=200)
# plt.plot(spikes.t / second, spikes.i, '.k', ms=0.7)

# show()

#%%

# Synaptically released glutamate 

fig, (ax1, ax2, ax3) = plt.subplots(3, 1,figsize=[10,12]) # Added figsize for better viewing




ax1.plot(traceS.t / second, traceS[0].Y_S, 'k', linewidth=0.7,label='[Glu]')
ax1.plot(spikes.t/second,spikes[0].i+3,'.g', ms=5,label='Spikes')
ax1.set_ylabel('Mol')
ax1.legend()


ax2.plot(traceS.t / second, traceS[1].Y_S, 'k', linewidth=0.7)
ax2.plot(spikes.t/second,spikes[1].i+3,'.g', ms=5)
ax2.set_ylabel('Mol')


ax3.plot(traceS.t / second, traceS[2].Y_S, 'k', linewidth=0.7)
ax3.plot(spikes.t/second,spikes[2].i+3,'.g', ms=5)
# fig.show()
ax3.set_ylabel('Mol')
ax3.set_xlabel('Time [s]')
# plt.figure(dpi=200)
# plt.plot(spikes.t / second, spikes.i, '.k', ms=0.7)

show()

#%%

# SYNAPTIC CURRENTS



fig, (ax1, ax2, ax3) = plt.subplots(3, 1,figsize=[10,12]) # Added figsize for better viewing



ax1.plot(trace.t/second, trace[0].I_syn, 'k', label='I_syn')
ax1.plot(trace.t/second, trace[0].I_ampa, 'r', label='I_ampa')
ax1.plot(trace.t/second, trace[0].I_nmda, 'b', label='I_nmda')
ax1.plot(spikes.t/second, spikes[0].i+1e-11, '.g', ms=5, label='Spikes') # Or a more descriptive label like 'Neuron Spikes'
ax1.set_ylabel('A')
ax1.legend()


ax2.plot(trace.t/second, trace[1].I_syn, 'k')
ax2.plot(trace.t/second, trace[1].I_ampa, 'r')
ax2.plot(trace.t/second, trace[1].I_nmda, 'b')
ax2.plot(spikes.t/second,spikes[1].i+1e-11,'.g', ms=5)
ax2.set_ylabel('A')


ax3.plot(trace.t/second, trace[2].I_syn, 'k')
ax3.plot(trace.t/second, trace[2].I_ampa, 'r')
ax3.plot(trace.t/second, trace[2].I_nmda, 'b')
ax3.plot(spikes.t/second,spikes[2].i+1e-11,'.g', ms=5)
# fig.show()
ax3.set_ylabel('A')
ax3.set_xlabel('Time [s]')
# plt.figure(dpi=200)
# plt.plot(spikes.t / second, spikes.i, '.k', ms=0.7)

show()









