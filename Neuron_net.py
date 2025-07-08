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
simtime = 10 * second               # simulation time
transient = 3 * second              # time omitted as transient
sed = 39                             # random number seed
devices.device.seed(sed)            # set the seed for all the random number realisations


# network parameters
Nl = 10
N = Nl * Nl                         # number of neurons
                     # connection probability between neurons
connprob = 0.2
distributed = True                  # Choose if the synaptic weights are normally distributed with standard deviation sd
sd = 0.7
DistDelays = False                   # Choose if the delays should be distant dependent
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
sigma = 5 * mV                      # standard deviation of the noisy voltage fluctuations

# Adaptation parameters
E_AHP = EK                          # Nernst potential of afterhyperpolarization current
g_AHP = 5 * nS                      # Maximum conductance of sAHP channels
tau_Ca = 8000 * ms                  # recovery time constant sAHP channels
alpha_Ca = 0.00035                  # strength of the spike-frequency adaptation

# synapse parameters
S = 0.4                             # Overall synaptic strength multiplicative factor
delta = 0.4                         # changes NMDAR/AMPAR ratio, should be between -1 and 1
# g_ampa = 0 * nS           # maximal conductance of AMPA channels
# g_nmda = 0 * nS           # maximal conductance of NMDA channels
g_ampa = (0.5 + delta) * nS           # maximal conductance of AMPA channels
g_nmda = (0.5 - delta) * nS           # maximal conductance of NMDA channels
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

# Make population of neurons
P = NeuronGroup(N, model=eqs, threshold='V>0*mV', reset='Ca += alpha_Ca', refractory=2 * ms,
                    method='exponential_euler')

# Initialize neuron parameters
P.V = -39 * mV                          # approximately resting membrane potential
P.I = '(rand() -0.5) * 10 * pA'          # Make neurons heterogeneously excitable

# Position neurons on a grid
grid_dist = 45 * umeter
P.x = '(i % Nl) * grid_dist'
P.y = '(i // Nl) * grid_dist'

# synapse model
if AsynchronousRelease:
    eqs_synapsmodel = '''
    s_nmda_tot_post = w * S * s_nmda * x_d :1 (summed) 
    qar_tot_post = w * S * x0 * qar :Hz (summed)
    qar = clip(randn()*sqrt(x_d/x0*uar*dt*(1-uar*dt))+uar*dt*x_d/x0, 0, 2*x_d/x0*uar*dt)/dt :Hz (constant over dt)
    ds_nmda/dt = -s_nmda/(taus_nmda)+alpha_nmda*(x_nmda)*(1-s_nmda) + x0 * qar : 1 (clock-driven)
    dx_nmda/dt = -x_nmda/(taux_nmda) :1 (clock-driven)
    dx_d/dt = (1-x_d)/tau_d -qar :1 (clock-driven)
    duar/dt = -uar/tau_ar :Hz (clock-driven)
    w : 1
    '''
    eqs_onpre = '''
    x_nmda += 1 
    x_d *= (1-U) 
    uar += Uar*(Umax-uar)
    s_ampa += w * S * x_d 
    '''
elif STF:
    eqs_synapsmodel = '''
    s_nmda_tot_post = w * S * x_d * u_d * s_nmda  :1 (summed)
    ds_nmda/dt = -s_nmda/(taus_nmda)+alpha_nmda*x_nmda*(1-s_nmda) : 1 (clock-driven)
    dx_nmda/dt = -x_nmda/(taux_nmda) :1 (clock-driven)
    dx_d/dt = (1-x_d)/tau_d :1 (clock-driven)
    du_d/dt = -u_d/tau_f :1 (clock-driven)
    w : 1
    '''
    eqs_onpre = '''
    x_nmda += 1
    x_d *= (1-u_d)
    u_d += U*(1-u_d)
    s_ampa += w * S * x_d * u_d
    '''
else:
    eqs_synapsmodel = '''
    s_nmda_tot_post = w * S * x_d * s_nmda  :1 (summed)
    ds_nmda/dt = -s_nmda/(taus_nmda)+alpha_nmda*x_nmda*(1-s_nmda) : 1 (clock-driven)
    dx_nmda/dt = -x_nmda/(taux_nmda) :1 (clock-driven)
    dx_d/dt = (1-x_d)/tau_d :1 (clock-driven)
    w : 1
    '''
    eqs_onpre = '''
    x_nmda += 1
    x_d *= (1-U)
    s_ampa += w * S * x_d 
    '''

# Make synapses
Conn = Synapses(P, P, model=eqs_synapsmodel, on_pre=eqs_onpre, method='euler')

# connect neurons
Conn.connect(p=connprob)

if distributed:
    Conn.w[:] = 'clip(1.+sd*randn(), 0, 2)'
else:
    Conn.w[:] = 1

Conn.x_d[:] = 1

if DistDelays:
    Vmax = (sqrt(Nl ** 2 + Nl ** 2) * grid_dist) / Maxdelay
    Conn.delay = '(sqrt((x_pre - x_post)**2 + (y_pre - y_post)**2))/Vmax'
else:
    Conn.delay = Maxdelay

# SET UP MONITORS AND RUN
recordstring = ['V']


dt2 = defaultclock.dt                                           # Allows for chanching the timestep of recording

trace = StateMonitor(P, recordstring, record=True, dt=dt2)
#%%
import random
recordlist = random.sample(range(1, len(Conn)-1), 100)          # Only measure a subset of synapses for speed


spikes = SpikeMonitor(P)

run(simtime, report='text', profile=True)

# PLOTS

plt.figure(dpi=200)
plot(trace.t / second, trace[2].V / mV, 'k', linewidth=0.7)
xlabel('time (s)')
ylabel('Membrane Potential of a neurons (mV)')


show()


plt.figure(dpi=200)
plt.plot(spikes.t / second, spikes.i, '.k', ms=0.7)
xlabel('time (s)')
ylabel('neuron index')
#xlim([45, 49])
#xlim([transient/second, simtime/second])

show()

