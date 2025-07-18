"

import matplotlib.pyplot as plt
from brian2 import *
import numpy as np
start_scope()


# simulation parameters
simtime = 100 * second               # simulation time
transient = 3 * second              # time omitted as transient
sed = 39                             # random number seed
devices.device.seed(sed)            # set the seed for all the random number realisations


# --- Parameters ---

def Get_efficacy_scale(td,scaling_f,rho,Y_T):
    
    '''
    This function calculates the Syanptic efficacy scaling factor. It is used to 
    scale the post-synaptic currents gating variable.
        
    
    Params:
        td = decay time scale
        rho = Vescicular versus mixing volume ratio
        Y_T = Total vescicular glutamate concentration
    '''
    
    return scaling_f / (rho * Y_T * td)


def Get_amplitude_scale(tr,td,scaling_f):
    
    '''
    This function calculates the Syanptic amplitude scaling factor. It is used to 
    scale the post-synaptic currents amplitude
    
    Params:
        td = decay time scale
        tr = rise time scale
       
    '''

    rise_ratio = tr / (td - tr)
    decay_ratio = td / (td - tr)
    Numerator = scaling_f * ((1 / td) - (1 / tr))
    Denominator = ((tr / td) ** decay_ratio) - ((tr / td) ** rise_ratio)
    
    Scaling_f_I = Numerator / Denominator


    return Scaling_f_I













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


# --- Neuron Model ---
#%
# ---------------------- NEURONAL GROUP ----------------------

# Variables 


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


Syn_model = 'Nin'


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
if Syn_model == 'Nina':

        
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
                         
                         
                            # r_SS_ampa = (alpha_ampa_new * Y_S)/(alpha_ampa_new * Y_S + beta_ampa_new) : 1
                            # r_SS_nmda = (alpha_nmda_new * Y_S)/(alpha_nmda_new * Y_S + beta_nmda_new ) : 1
                            # tau_ampa = 1/(alpha_ampa_new * Y_S + beta_ampa_new ) : second
                            # tau_nmda = 1/(alpha_nmda_new * Y_S + beta_nmda_new ) : second
                               
                            dr_ampa/dt = alpha_ampa_new * Y_S * (1 - r_ampa) - beta_ampa_new * r_ampa: 1 (clock-driven)
                            dr_nmda/dt =  alpha_nmda_new * Y_S * (1 - r_nmda) - beta_nmda_new * r_nmda : 1 (clock-driven)
                            
                            # dr_ampa/dt =  -beta_ampa_new* r_ampa: 1 (clock-driven)
                            # dr_nmda/dt =  - beta_nmda_new * r_nmda : 1 (clock-driven)
                            
                            r_ampa_tot_post = r_ampa : 1 (summed)
                            r_nmda_tot_post = r_nmda : 1 (summed)
                         
                         
                        ''')
                        
                        
    # pre += '''           
    #         r_ampa += alpha_ampa_new * Y_S * (1 - r_ampa)
    #         r_nmda +=  alpha_nmda_new * Y_S * (1 - r_nmda)
           
    #                '''          

    
    
    eqs_NN += Equations(''' 
                        I_syn =  I_ampa + I_nmda: amp
                        I_ampa = g_ampa*(V-E_ampa)*(r_ampa_tot) : amp
                        I_nmda = g_nmda*(V-E_nmda)*(r_nmda_tot)/(1+exp(-0.062*V/mV)/3.57) : amp
                        r_nmda_tot :1
                        r_ampa_tot :1
                        
                    
                        ''')
    
    


params_NN = get_Neuronparam(sigma = 0*mV)
params_Syn = get_Synparam(Y_T = 300.*mmole,alpha_ampa_new = 1.1e4 * 1/mole * 1/second,beta_ampa_new = 800 * 1/second )



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
#
# --- Run Simulation ---
run(simtime,report='text',profile=True)
 #%%
 
 # --- Plotting Results ---
figure(figsize=(10, 6))

subplot(3, 1, 1)
plot(spike_monitor_input.t/ms, spike_monitor_input.V, '.k')
title('Random Input Spike Train')
xlabel('Time (ms)')
ylabel('Input Neuron Index')
xlim(0, spike_train_duration/ms)

subplot(3, 1, 2)
plot(state_monitor.t/ms, state_monitor.v[0]/mV)
title('Neuron Membrane Potential')
xlabel('Time (ms)')
ylabel('Voltage (mV)')
xlim(0, spike_train_duration/ms)

subplot(3, 1, 3)
plot(spike_monitor_neuron.t/ms, spike_monitor_neuron.i, '.r')
title('Output Neuron Spikes')
xlabel('Time (ms)')
ylabel('Neuron Index')
xlim(0, spike_train_duration/ms)
ylim(-0.5, 0.5) # To make the single spike clearly visible
yticks([]) # Remove y-ticks for single neuron

tight_layout()
show()
#%%
Is = state_monitorsynapse.Y_S
IY_S = synapse.Y_S.unit
rnmda = state_monitorsynapse[0].r_nmda
raamp = state_monitorsynapse[0].r_ampa
#%%
plt.figure()
plt.plot(spike_monitor_input.t / second, spike_monitor_input.i, '.k', ms=5)
xlim(0, 10/ms)
show()

#%%
plt.figure()
plt.plot(state_monitor.t / second, state_monitor[0].V, 'k', ms=5)

xlim(0, 1)
show()

#%%


plt.figure(dpi=200)
plt.plot(state_monitorsynapse.t /ms, state_monitorsynapse[0].Y_S, 'k')
xlim(0, 1000)
plt.ylabel(f'{IY_S}')
show()
#%%
# %matplotlib
figure(figsize=(10, 6))
Iunit = neuron.I_syn.unit
plt.plot(spike_monitor_input.t / ms, spike_monitor_input.i+1e-11, '.g', ms=5)
plt.plot(state_monitor.t/ms, state_monitor[0].I_syn, 'k') 
plt.plot(state_monitor.t/ms, state_monitor[0].I_ampa, 'r')
plt.plot(state_monitor.t/ms, state_monitor[0].I_nmda, 'b')
plt.title('Synaptic currents')
plt.xlabel('Time (ms)')

plt.ylabel(f'{Iunit}')


plt.xlim((0,1/ms))

plt.show()


#%%
figure(figsize=(10, 6))
Iunit = neuron.I_syn.unit
# plt.plot(spike_monitor_input.t / ms, spike_monitor_input.i+1e-11, '.g', ms=5)
plt.plot(state_monitor.t/ms, state_monitorsynapse[0].r_nmda, 'k')
plt.plot(state_monitor.t/ms, state_monitorsynapse[0].r_ampa, 'r')
# plt.plot(state_monitor.t/ms, state_monitor[0].I_ampa, 'r')
# plt.plot(state_monitor.t/ms, state_monitor[0].I_nmda, 'b')
plt.title('Synaptic currents')
plt.xlabel('Time (ms)')
plt.ylabel(f'{Iunit}')

plt.xlim((0,1000))

plt.show()
#%%
figure(figsize=(10, 6))
Iunit = neuron.I_syn.unit
# plt.plot(spike_monitor_input.t / ms, spike_monitor_input.i+1e-11, '.g', ms=5)
plt.plot(state_monitor.t/ms, state_monitorsynapse[0].r_SS_ampa, 'k')
plt.plot(state_monitor.t/ms, state_monitorsynapse[0].r_SS_nmda, 'r')
# plt.plot(state_monitor.t/ms, state_monitor[0].I_ampa, 'r')
# plt.plot(state_monitor.t/ms, state_monitor[0].I_nmda, 'b')
plt.title('Synaptic currents')
plt.xlabel('Time (ms)')
plt.ylabel(f'{Iunit}')

plt.xlim((0,1000))

plt.show()
