
from brian2 import *

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
    
    N_params = get_Neuronparam()
    
    coordinates = []
    for _ in range(N):
        x = random.uniform(N_params['c_min'], N_params['c_max'])  # Generates a float between c_min (inclusive) and c_max (exclusive)
        y = random.uniform(N_params['c_min'], N_params['c_max'])
        coordinates.append((x, y))
    return np.array(coordinates)

def get_Neuronparam(**kwargs):
    params = {
        'a': 0.02, # The parameter 'a' describes the time scale of the recovery variable 'u'. Smaller values result in slower recovery.
        'b': 0.2,  # The parameter 'b' describes the sensitivity of the recovery variable 'u' to the subthreshold fluctuations of the membrane potential 'v'.
        'c': -65,  # The parameter 'c' describes the after-spike reset value of the membrane potential 'v'.
        'd': 8,    # The parameter 'd' describes the after-spike reset of the recovery variable 'u'.
        'I_inj': 5,# The parameter 'I_inj' represents the injected DC current.
        'sigma': 2,# The parameter 'sigma' represents the standard deviation of the injected noise.
        'c_min': 0, # The parameter 'c_min' defines the minimum coordinate for neuron placement.
        'c_max': 1100, # The parameter 'c_max' defines the maximum coordinate for neuron placement.
        # --- STD ---
        'tau_d': 813 * ms, # The recovery time constant of synaptic depression.
        'U': 0.2, # The magnitude of synaptic depression.
    }
    params.update(kwargs)
    return params

def get_Synapseparam(**kwargs):
    params = {
        # --- PSC kinetics ---
        'tau_decay_ampa': 2.4 * ms, # The decay time constant of the AMPA receptor-mediated postsynaptic current.
        'tau_decay_nmda': 100.0 * ms,# The decay time constant of the NMDA receptor-mediated postsynaptic current.
        'tau_rise_nmda': 2.0 * ms, # The rise time constant of the NMDA receptor-mediated postsynaptic current.
        'delta_ampa': 2, # The factor for AMPA receptor-mediated synaptic current.
        'delta_nmda': 1, # The factor for NMDA receptor-mediated synaptic current.
        'delta_gaba': 20, # The factor for GABA receptor-mediated synaptic current.
        'tau_decay_gaba': 2, # The decay time constant of the GABA receptor-mediated postsynaptic current.
        # --- Connectivity ---
        'connprob': 0.2, # The connection probability between neurons (random connectivity).
        'slope': 1/500, # The slope for distance-dependent connection probability.
        'intercept': 0.8, # The intercept for distance-dependent connection probability.
    }
    params.update(kwargs)
    return params



def get_Astrocyteparams(**kwargs):
    params = {
        'Omega_IP3': 152.3, # IP3 degradation rate [1/s]
        'Omega_Ca': 0.05, # Accumulation rate between IP3 and Ca.
        'Omega_GT': 0.077, # Recovery rate of GT receptors [1/s]
        'GT_r': 0.3, # Fraction of unbound receptors recruited by gliotransmission
        'Ca_TH': 0.1, # Calcium threshold for gliotransm. released
        'alpha_astro': 0.7, # Influence of gliatransmittion on pre-syn release
        'y_astro': 0.01 # Astrocytic-driven hyperpolarizing current amplitude
    }
    params.update(kwargs)
    return params
    
    
    



def Neuronal_Network(Nn,I_inj,Connection_var,sigma_noise,sed,**kwargs):
    
    
    NN_eqs = Equations('''
                        dv/dt = (0.04*v**2 + 5*v + 140 - u + I + I_syn)/ms + sigma*xi/ms**0.5: 1
                        du/dt = a*(b*v - u)/ms : 1
                        dx_d/dt = (1-x_d)/tau_d :1
                        I : 1
                        x : meter
                        y : meter
                        ''')
                        
    params_NN = get_Neuronparam(I_inj = I_inj,sigma=sigma_noise)                    
                        
    
                  
    # ---------- SYNAPSES ---------- 
   
    params_Syn = get_Synapseparam()
   

    # --- Excitatory group ---                  
                         
    eqs_Syn = Equations('''
                         
                               
                            dr_ampa/dt = -r_ampa/tau_decay_ampa : 1 (clock-driven)
     
                            dr_nmda/dt = ((tau_decay_nmda / tau_rise_nmda) ** (tau_rise_nmda / (tau_decay_nmda - tau_rise_nmda))*x_r_nmda-r_nmda)/tau_rise_nmda : 1 (clock-driven)
                            dx_r_nmda/dt = -x_r_nmda/tau_decay_nmda   : 1 (clock-driven)
                 
                            I_ampa_post = r_ampa : 1 (summed)
                            I_nmda_post = r_nmda : 1 (summed)
                         
                         
                        ''')
                        
                        
    pre_syn = '''           
            r_ampa   +=  delta_ampa*x_d_pre
            x_r_nmda +=  delta_nmda*x_d_pre
           
                    '''          
     
    
    
    NN_eqs += Equations(''' 
                        I_syn =  I_ampa + I_nmda : 1
                        I_nmda : 1 
                        I_ampa : 1
                        
                    
                        ''')     
     
                        
     
        
     
    # ---------- INITIALIZE GROUPS ----------
                        
    N = NeuronGroup(Nn, NN_eqs,
                    name='Neuron*',
                    namespace= params_NN,
                    refractory= 2 * ms,
                    threshold='v > 30',
                    reset='v = c; u += d;x_d *= (1-U)',
                    method='euler')                  
    # ----- SET POSITIONS AND CONNECTIONS -----
    # Position neurons on a grid

    Coordinates = get2D_rnd_coordinates(N.N,c_min,c_max,sed)
    N.x = Coordinates[:,0]*um
    N.y = Coordinates[:,1]*um                     
                        
    # Initialize neuron parameters
    N.v = -39                          # approximately resting membrane potential
    N.I = '(rand() -0.5) * I_inj'          # Make neurons heterogeneously excitable                    
                        
                        
                        
                  
                        
    S = Synapses(N,N, model=eqs_Syn,
                        on_pre=pre_syn,
                        name='Synapse*',
                        namespace=params_Syn,
                        method='exponential_euler')
                         
    
    # Check the type of connection scheme and implement the connection
    if isinstance(Connection_var, str):
        
        
        if Connection_var == 'Random':
            # --- Random ---
            S.connect(p=params_Syn['connprob'])
            
        elif Connection_var == 'Distance':
            
            ADJ_ = Distance_based_connections(N,params_Syn,sed)
            
                
            Source_neuron,Target_neuron = unfold_ADJ(ADJ_)
            
            S.connect(i=Source_neuron, j=Target_neuron)
            
            os.chdir(Out_path)
            # Save the array to a CSV file named 'my_data.csv'
            np.savetxt('ADJ.csv', ADJ_, delimiter=',')
            
            
      
        
    elif isinstance(Connection_var, numpy.ndarray):
        
    
    
        Source_neuron,Target_neuron = unfold_ADJ(Connection_var)
        
        S.connect(i=Source_neuron, j=Target_neuron)
    
    
    
    return N,S
    
      
    
    
    
    
    
