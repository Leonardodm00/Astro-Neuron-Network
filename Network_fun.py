# -*- coding: utf-8 -*-
"""
Created on Tue Jul  8 11:50:21 2025

@author: leona
"""

'''
This script works on Brian2 simulator of neuronal networks.

Some remarks:
    
    - Within and between connections among neuronal and astrocytic networks are defined outside.
      
    - Astrocytes are masked under the Neuron class of Brian and the gliotransmission is 
      implemented as a specific type of synapse.



REFERENCES:


    - Neuronal network: An in silico and in vitro human neuronal network model reveals cellular 
                        mechanisms beyond NaV1.1 underlying Dravet syndrome
    
    - Synapses: Modulation of Synaptic Plasticity by Glutamatergic Gliotransmission: A Modeling Study
        
    - Astrocyte network: Modulation of Synaptic Plasticity by Glutamatergic Gliotransmission: A Modeling Study

        
'''

from brian2 import *
# Define some global lambdas
peak_normalize = lambda peak, taur, taud : peak*(1./taud - 1./taur)/Hz/( (taur/taud)**(taud/(taud-taur))-(taur/taud)**(taur/(taud-taur)) )




#%%
# ------------------- DEFINE THE PARAMETERS -------------------

def get_parameters(oscillations='AM', synapse='depressing', adaptation = True,**kwargs):

    # Parameters from Tables 1-3, using Table 2 if the values differ from Table 1
    # Common parameters
    
    Neuron_area = 300*umetre**2
    
    
    parameters = {
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
        
        
        # --- Neuron Parameters
       'area': Neuron_area,               # membrane area of the neuron
       'Cm': (2*ufarad*cm**-2) * Neuron_area, # membrane capacitance (calculated with area)
       'El': -39.2 * mV,                    # Nernst potential of leaky ions
       'EK': -80 * mV,                      # Nernst potential of potassium
       'ENa': 70 * mV,                      # Nernst potential of sodium
       'g_na': 1.6 * 50 * msiemens * cm**-2 * Neuron_area, # maximal conductance of sodium channels (calculated with area)
       'g_kd': 1.3 * 5 * msiemens * cm**-2 * Neuron_area,  # maximal conductance of potassium (calculated with area)
       'gl': (0.3*msiemens*cm**-2) * Neuron_area, # maximal leak conductance (calculated with area)
       'g_m': (0*msiemens*cm**-2) * Neuron_area, # maximal conductance of AHP currents
       'VT': -30.4*mV,                      # alters firing threshold of neurons
       'sigma': 6 * mV,                     # standard deviation of the noisy voltage fluctuations
       'Tau_max': 4000 * ms,                # Decay factor of AHP
    
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


    # ------------------ ASTROCYTES ------------------

    if oscillations == 'AM':
        parameters.update({
            'K_P': 0.1*umole,
            'O_delta': 0.01*umole/second,
            'Omega_5P': 0.1/second,
            'I_bias': 0.8*umole
        })
    elif oscillations == 'FM':
        parameters.update({
            'K_P': 0.05*umole,
            'O_delta': 0.05*umole/second,
            'Omega_5P': 0.1/second,
            'I_bias': 1.*umole
        })
    else:
        raise ValueError('oscillations argument has to be "AM" or "FM"')


    # ------------------ SYNAPSES ------------------
    if synapse == 'depressing':
        parameters.update({
            'Omega_d': 2./second,
            'Omega_f': 3.33/second,
            'U_0__star': 0.6,
            'alpha': 0.,
        })
    elif synapse == 'facilitating':
        parameters.update({
            'Omega_d': 2./second,
            'Omega_f': 2./second,
            'U_0__star': 0.15,
            'alpha': 1.,
        })
    elif synapse == 'neutral':
        parameters.update({
            'Omega_d': 3./second,
            'Omega_f': 3./second,
            'U_0__star': 0.5,
            'alpha': 1.,
        })
    else:
        raise ValueError('synapse argument has to be "depressing", "facilitating" or "neutral"')

    # Post-synaptic neuron parameters
    parameters.update({
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
        'we_AMPA' : 0.5, # Relative contribution of AMPA channels to the total syn weight
        'we_NMDA' : 0.5, # Relative contribution of NMDA channels to the total syn weight
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
    parameters.update({
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
    
    
    
    # ------------------ NEURONS ------------------
    
    if adaptation == True:
        
        parameters.update({
            
            'g_m':(0.4*msiemens*cm**-2) * parameters('area')
                           })
        
    
        
        
    
    
    
    parameters.update(kwargs)
    
    
    
    return parameters







def neuron(params):
    # This model must be match with the TK synaptic model
    
    # neuron model
    eqs = Equations('''
    dV/dt = noise + (-gl*(V-El)-g_na*(m*m*m)*h*(V-ENa)-g_kd*(n*n*n*n)*(V-EK)+I-I_syn+I_AHP)/Cm : volt
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
    noise = sigma*(2*gl/Cm)**.5*randn()/sqrt(dt) :volt/second (constant over dt)
    I : amp
    I_syn =  I_ampa + I_nmda: amp
    I_ampa = g_ampa*(V-E_ampa)*(s_ampa) : amp
    I_nmda = g_nmda*(V-E_nmda)*(s_nmda_tot)/(1+exp(-0.062*V/mV)/3.57) : amp
    s_nmda_tot :1
    ds_ampa/dt = -s_ampa/tau_ampa + qar_tot :1 
    qar_tot : Hz
    I_AHP = -g_AHP*Ca*(V-E_AHP) : amp
    dCa/dt = - Ca / tau_Ca : 1 
    x : meter
    y : meter
    ''')





def synapse(source, target, params, positions ,connect='i==j', ics=None, dt=None,
            name='synapses*',postc_sic=None, sic=None, delay=None,
            RandomKinetics = False, OnlyExc = True):
    
    
    
    '''
    
    delay: 3 element array. [0] : If to use it, [1] : Type of delay 'random', 'distance'.
                            [2] : Max_delay.
    
    positions: position of neuronal cells.
    
    
    '''
    
    
    # -------------- Equations --------------
    
    # Synapses modelled as in Tsodyks (2005) with basal release probability
    # modulated by presynaptic receptors as in De Pitta' et al., PLoS Comput. Biol. (2011)
    #
    # IMPORTANT: 'postc' argument stands for 'post' in other methods, but because 'post' is a protected keyword in Synapse
    # it cannot be used in this module and 'postc' is used instead.

    eqs = Equations('''
        # Fraction of activated presynaptic receptors
        dGamma_S/dt = O_G * G_A * (1 - Gamma_S) - Omega_G * Gamma_S : 1
        # Usage of releasable neurotransmitter per single action potential:
        du_S/dt = -Omega_f * u_S : 1 (event-driven)
        # Fraction of synaptic neurotransmitter resources available for release:
        dx_S/dt = Omega_d *(1 - x_S) : 1 (event-driven)
        dY_S/dt = -Omega_c * Y_S : mole
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
    U_0 = (1 - Gamma_S) * U_0__star + alpha * Gamma_S
    u_S += U_0 * (1 - u_S)
    r_S = u_S * x_S # released synaptic neurotransmitter resources
    x_S -= r_S
    Y_S += rho_c * Y_T * r_S
    '''
    post = None

    # -------------- Currents --------------
    # Modelled both AMPA and NMDA and GABA
    # Also the weights are added
    
    if RandomKinetics == True:
        # Some variance is given to the time constant of synaptic transmission
        
        # Determine the number of synapses
        N_syn = len(source)
        
        if OnlyExc == True:
            
            AMPA_rise = np.random.normal(loc=params['AMPA_tau_e_r'], scale=params['AMPA_tau_e_r_STD'], size=N_syn)
            NMDA_rise = np.random.normal(loc=params['NMDA_tau_e_r'], scale=params['NMDA_tau_e_r_STD'], size=N_syn)
            AMPA_decay = np.random.normal(loc=params['AMPA_tau_e'], scale=params['AMPA_tau_e_STD'], size=N_syn)
            NMDA_decay = np.random.normal(loc=params['NMDA_tau_e'], scale=params['NMDA_tau_e_STD'], size=N_syn)
            we = np.random.normal(loc=params['we'], scale=params['we_std'], size=N_syn)
            
            
            # JUST AMPA USED
            parameters.update({  
                
                'AMPA_tau_e_r': AMPA_rise,
                'NMDA_tau_e_r': NMDA_rise,
                'AMPA_tau_e': AMPA_decay,
                'NMDA_tau_e': NMDA_decay,
                'we':we
                
                })
                    
                
                
                
                
                
            
            
            
            
            
        else:
            
            N_syn_exc = np.int16(N_syn * params['E_Iper'])
            N_syn_inh = N_syn - N_syn_exc
            
            
            AMPA_rise = np.random.normal(loc=params['AMPA_tau_e_r'], scale=params['AMPA_tau_e_r_STD'], size=N_syn_exc)
            NMDA_rise = np.random.normal(loc=params['NMDA_tau_e_r'], scale=params['NMDA_tau_e_r_STD'], size=N_syn_exc)
            AMPA_decay = np.random.normal(loc=params['AMPA_tau_e'], scale=params['AMPA_tau_e_STD'], size=N_syn_exc)
            NMDA_decay = np.random.normal(loc=params['NMDA_tau_e'], scale=params['NMDA_tau_e_STD'], size=N_syn_exc)
            
            GABA_rise = np.random.normal(loc=params['GABA_tau_i_r'], scale=params['GABA_tau_i_r_STD'], size=N_syn_inh)
            GABA_decay = np.random.normal(loc=params['GABA_tau_i'], scale=params['GABA_tau_i_STD'], size=N_syn_inh)
            
            we = np.random.normal(loc=params['we'], scale=params['we_std'], size=N_syn_exc)
            wi = np.random.normal(loc=params['we'], scale=params['we_std'], size=N_syn_inh)
            
            
            ###### TO FINISH
        
        
        
        
        
    
    
    
    '''
    
    ADD THE NMDA AND AMPA and GABA
    
    
    
    '''

    # Add case-specific code
    if postc_sic =='exp' :
        eqs += Equations('''
                         dge_AMPA/dt = -ge_AMPA/AMPA_tau_e + Y_S* we/AMPA_tau_e : 1
                        
                         ''')
    elif postc_sic =='double-exp':
        params['nge_AMPA'] = peak_normalize(1.,params['AMPA_tau_e_r'],params['AMPA_tau_e'])
        params['nge_NMDA'] = peak_normalize(1.,params['NMDA_tau_e_r'],params['NMDA_tau_e'])
        eqs += Equations('''
                       dge_AMPA/dt = -ge_AMPA/AMPA_tau_e_r + nge_AMPA*B_syn_AMPA : 1
                       dB_syn_AMPA/dt = -B_syn_AMPA/AMPA_tau_e + Y_S* (1 - we_AMPA)*we : Hz  # Technically you need the concentration in the cleft, but you can use Y_S and rescale w/in 'w'
                       
                       dge_NMDA/dt = -ge_NMDA/NMDA_tau_e_r + nge_NMDA*B_syn_NMDA : 1
                       dB_syn_NMDA/dt = -B_syn_NMDA/NMDA_tau_e + Y_S* (1 - we_NMDA)*we : Hz  # Technically you need the concentration in the cleft, but you can use Y_S and rescale w/in 'w'
                       ''')


    if sic==True:
        if postc_sic =='exp':
            eqs += Equations('''
                           dgsic/dt = -gsic/tau_sic + G_A_sic*wa/tau_sic : 1
                           G_A_sic : mole
                           ''')
        elif postc_sic =='double-exp':
            params['nsic'] = peak_normalize(1.,params['tau_sic_r'],params['tau_sic'])
            eqs += Equations('''
                           dgsic/dt = -gsic/tau_sic_r + nsic*B_sic : 1
                           dB_sic/dt = -B_sic/tau_sic + G_A_sic*wa : Hz  # Technically you need the concentration in the cleft, but you can use Y_S and recale w/in 'we'
                           G_A_sic : mole
                           ''')


    




    #!!! TO CHECK
    synapses = Synapses(source, target, eqs,
                        on_pre=pre,
                        on_post=post,
                        namespace=params,
                        name=name,
                        dt=dt)
    
    
    
    # -------------- Connections --------------
    synapses.connect(connect)
    
    # --- Delays ---
    if delay[0] == True:
        
        Max_delay = delay[2] # Max conduction delay expressed in ms
        N_synapses = synapses.N
        
        if delay[1] == 'random':
            # Randomly generate scaling values.
            random_values = np.random.rand(N_synapses)
            Delays = (Max_delay * ms) * random_values
            synapses.delay = Delays
            
        elif delay[1] == 'distance':    
            # Retrieve the distances between neuronal somata.
            # Unmyelinated nerve fibers. Usually reach 0.5 [mm/ms] speed.
            # To this it must be add the time required from the vescicle release 
            # at the pre-synaptic site.
            random_values = np.random.rand(N_synapses)
            Delays = (Max_delay * ms) * random_values
            Conductance_velocity = 0.5 * (mm/ms) + Delays
            synapses.delay = '(sqrt((x_pre - x_post)**2 + (y_pre - y_post)**2 + (z_pre - z_post)**2)) * Conductance_velocity'

 


        
    # -------------- Initialization --------------
    synapses.x_S = 1.0

    

    # Random initialization of initial conditions
    if ics=='rand':
        synapses.u_S = 'rand()'
        synapses.x_S = 'rand()'
        Y_T = params['Y_T']
        synapses.Y_S = '1.2 * rho_c * Y_T * rand()'

    return synapses


def astrocyte_group(N, params, dt=1*msecond, ics=None):
    
    
    
    
    # -------------- Equations --------------
    
    eqs = '''
    # Fraction of activated astrocyte receptors:
    dGamma_A/dt = O_N * (Y_bias+Y_extra)**n * (1 - Gamma_A) -
                  Omega_N*(1 + zeta * C/(C + K_KC)) * Gamma_A : 1

    # IP_3 dynamics:
    dI/dt = O_beta * Gamma_A + O_delta/(1 + I/K_delta) * C**2/(C**2 + K_delta**2) -
            O_3K * C**4/(C**4 + K_D**4) * I/(I + K_3K) - Omega_5P*I +
            I_coupling + I_exogenous : mole

    # Exogenous stimulation applied to the cell
    delta_I_bias = I - I_bias: mole
    I_exogenous = -F/2*(1 + tanh((abs(delta_I_bias) - I_Theta)/omega_I))*sign(delta_I_bias) : mole/second
    # I_exogenous : mole/second
    # diffusion between astrocytes:
    I_coupling : mole/second

    # Ca^2+-induced Ca^2+ release:
    dC/dt = (Omega_C * m_inf**3 * h**3 + Omega_L) * (C_T - (1 + rho_A)*C) -
            O_P * C**2/(C**2 + K_P**2) : mole
    dh/dt = (h_inf - h)/tau_h : 1  # IP3R de-inactivation probability
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
    '''
    
    # The definition of a threshold and reset mechanism in the astrocyte group
    # allows to use SpikeMonitors to estimate the frequency of oscillations
    group = NeuronGroup(N, eqs,
                        threshold='C>C_osc',
                        refractory='C>C_osc',
                        method='rk4',
                        dt=dt,
                        namespace=params,
                        name='astrocyte*')

    # Random initialization of initial conditions
    if ics=='rand':
        group.Gamma_A = 'rand()'
        group.I = '3*rand()*umole'
        group.C = '1.5*rand()*umole'
        group.h = 'rand()'

    return group

