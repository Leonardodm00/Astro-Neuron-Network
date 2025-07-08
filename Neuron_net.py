# -*- coding: utf-8 -*-
"""
Created on Tue Jul  8 14:40:44 2025

@author: Admin
"""

import matplotlib.pyplot as plt
from brian2 import *

# Ensure a clean Brian2 state for repeatable simulations
start_scope()

# SET PARAMETERS
# simulation parameters
simtime = 65 * second               # simulation time
transient = 5 * second              # time omitted as transient
sed = 39                             # random number seed
devices.device.seed(sed)            # set the seed for all the random number realisations

outputdir = '/home/Nina/Documents/FB_project/Output'    # directory to save figures
simname = '/test_'                                      # simulation name to save figures

# network parameters
Nl = 1
N = Nl * Nl                         # number of neurons
                     # connection probability between neurons

distributed = True                  # Choose if the synaptic weights are normally distributed with standard deviation sd
sd = 0.7
DistDelays = True                   # Choose if the delays should be distant dependent
Maxdelay = 25*ms                    # with maximum conduction delay

# neuron parameters
area = 300*umetre**2                # membrane area of the neuron
Cm = (2*ufarad*cm**-2) * area       # membrane capacitance
El = -39.2 * mV                     # Nernst potential of leaky ions
EK = -80 * mV                       # Nernst potential of potassium
ENa = 70 * mV                       # Nernst potential of sodium
g_na = 1.6 * 50 * msiemens * cm ** -2 * area  # maximal conductance of sodium channels
g_kd = 1.3 * 5 * msiemens * cm ** -2 * area   # maximal conductance of potassium
gl = (0.3*msiemens*cm**-2) * area   # maximal leak conductance
VT = -30.4*mV                       # alters firing threshold of neurons
sigma = 6 * mV                      # standard deviation of the noisy voltage fluctuations

# Adaptation parameters
E_AHP = EK                          # Nernst potential of afterhyperpolarization current
g_AHP = 5 * nS                      # Maximum conductance of sAHP channels
tau_Ca = 8000 * ms                  # recovery time constant sAHP channels
alpha_Ca = 0.00035                  # strength of the spike-frequency adaptation

# synapse parameters
S = 0.4                             # Overall synaptic strength multiplicative factor
delta = 0.6                         # changes NMDAR/AMPAR ratio, should be between -1 and 1
g_ampa = (1 + delta) * nS           # maximal conductance of AMPA channels
g_nmda = (1 - delta) * nS           # maximal conductance of NMDA channels
E_ampa = 0 * mV                     # Nernst potentials of synaptic channels
E_nmda = 0 * mV
tau_ampa = 2 * ms                   # recovery time constant of AMPA conductance
taus_nmda = 100 * ms                # rise time constant of NMDA conductance
taux_nmda = 2 * ms                  # decay time constant of NMDA conductance
alpha_nmda = 0.5 * kHz

tau_d = 200 * ms                    # Recovery time constant of short-term synaptic depression (STD)
U = 0.2                             # Strength of STD

STF = False                         # Whether to include Short-term synaptic facilitation (STF)
tau_f = 1000 * ms                   # Recovery time constant of STF

AsynchronousRelease = False         # Whether to include asynchronous neurotransmitter release
tau_ar = 700 * ms                   # Recovery time constant of asynchronous release
Uar = 0.003                         # increment of asynchronous release probability induced by an action potential
Umax = 0.5/ms                       # Saturation level of facilitation
x0 = 5                              # quantum size

# BUILD NETWORK
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


# Make population of neurons
P = NeuronGroup(N, model=eqs, threshold='V>0*mV', reset='Ca += alpha_Ca', refractory=2 * ms,
                    method='exponential_euler')

# Initialize neuron parameters
P.V = -39 * mV                          # approximately resting membrane potential
P.I =  0*nA         # Make neurons heterogeneously excitable




# --- Monitors ---
# Set up monitors to record the neuron's activity during the simulation.
M_v = StateMonitor(P, 'V', record=0) # Record membrane potential of neuron 0
M_gates = StateMonitor(P, ['m', 'h', 'n'], record=0) # Record gating variables

# --- Simulation Run ---
print("Starting simulation...")

# 1. Run for a short period to let the neuron settle at resting potential
print("Running initial settling period (50 ms)...")
run(50 * ms)


# 1. Run for a short period to let the neuron settle at resting potential
print("Running initial settling period (50 ms)...")
run(100 * ms)


# 2. Apply a constant injected current to trigger an action potential
print("Injecting current (20 nA for 50 ms)...")
P.I = 20 * pA
run(100 * ms)

# 3. Stop the injected current and observe the neuron's recovery
print("Current injection stopped. Observing recovery (50 ms)...")
P.I = 30 * pA
run(100 * ms)

print("Simulation finished.")
#%%
# --- Plotting Results ---

# Plot Membrane Potential (Voltage)
plt.figure(figsize=(10, 6))
plt.plot(M_v.t/ms, M_v.V[0]/mV)
plt.xlabel('Time (ms)')
plt.ylabel('Membrane Potential (mV)')
plt.title('Hodgkin-Huxley Neuron: Membrane Potential')
plt.grid(True)
plt.show()

# Plot Gating Variables
plt.figure(figsize=(10, 6))
plt.plot(M_gates.t/ms, M_gates.m[0], label='m (Na activation)')
plt.plot(M_gates.t/ms, M_gates.h[0], label='h (Na inactivation)')
plt.plot(M_gates.t/ms, M_gates.n[0], label='n (K activation)')
plt.xlabel('Time (ms)')
plt.ylabel('Gating Variable Value')
plt.title('Hodgkin-Huxley Neuron: Gating Variables')
plt.legend()
plt.grid(True)
plt.show()
