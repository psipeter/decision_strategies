import numpy as np
import nengo
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import nengolib
import scipy
sns.set(style="white", context="poster")

# Parameters 
validities = np.array([0.706, 0.688, 0.667, 0.647, 0.625, 0.6]) # validity of cues (from experiment)
T = 1.0  # time interval (s) for which the choices are presented
trial = 0
ramp = 0.4
seed = 0
noise = 5e-2
valence = 0
tau = 0.1

''' Read empirical data from .csv files '''
cues = pd.read_csv('how_many_cues.csv', sep=';')
stim = pd.read_csv('input_stimuli.csv',sep=';')
choices = pd.read_csv('choices.csv', sep=';')
def read_values(trial):
    trdata = list(stim[['c1','c2','c3','c4','c5','c6','c1.1','c2.1','c3.1','c4.1','c5.1','c6.1']].loc[trial])
    return trdata[:len(trdata)//2], trdata[len(trdata)//2:]
values_A, values_B = read_values(trial)

# Inputs and Functions

values = [[values_A[n], values_B[n]] for n in range(len(values_A))]
validity_process = nengo.processes.PresentInput(validities, presentation_time=T)
value_process = nengo.processes.PresentInput(values, presentation_time=T)
noise_process = nengo.processes.WhiteSignal(period=6, high=10, rms=noise, seed=seed)
time_process = lambda t: t*ramp + 10*(t>5.5)  # time pressure ramps over each trial
emotion_process = lambda t: valence  # fixed input for pre-trial viewing of arousing images
mult = lambda x: [x[0]*x[2], x[1]*x[2]]  # values weighted by remembered validity
delta = lambda x: np.abs(x[0] - x[1])  # decision conflict (difference in accum. evidence)
urgency = lambda x: [x[0]+x[1], x[0]+x[1]]  # total arousal (time+emotion) added to action utility
conflict = lambda x: [x[0], x[0]]  # greater delta means less conflict, increases utility, effectively lowers thr

''' Model definition '''

model = nengo.Network(seed=seed)
with model:
    # Inputs
    reward_memory = nengo.Node(validity_process, label="reward memory")
    sensory_input = nengo.Node(value_process, label="sensory input")
    elapsed_time = nengo.Node(time_process, label="elapsed time")
    memory_noise = nengo.Node(noise_process, label="memory noise")
    emotional_state = nengo.Node(emotion_process, label="emotional state")
    # motor_output = nengo.Node(size_in=2, size_out=2)
    # validity_node = nengo.Ensemble(1, 3, neuron_type=nengo.Direct())  # compute in math
    # evidence_node = nengo.Ensemble(1, 2, neuron_type=nengo.Direct())  # compute in math
    
    # Ensembles
    raw_value = nengo.Ensemble(100, 2, radius=2, label="raw value (vision)")
    weighted_value = nengo.Ensemble(200, 3, radius=3, label="weighted value (OFC)")
    evidence = nengo.Ensemble(200, 2, radius=4, label="accum. evidence (PFC)")
    utility = nengo.Ensemble(200, 2, radius=4, label="action utility (pSMA)")
    task_monitor = nengo.Ensemble(200, 2, radius=4, label="task monitor (ACC)")
    arousal = nengo.Ensemble(200, 2, radius=3, label="arousal (LC)")
    decision = nengo.networks.BasalGanglia(dimensions=2, n_neurons_per_ensemble=100)
    motor_output = nengo.networks.Thalamus(dimensions=2, n_neurons_per_ensemble=100)
    
    # Connections
    # inputs
    nengo.Connection(sensory_input, raw_value, synapse=None)
    nengo.Connection(reward_memory, weighted_value[2], synapse=None)
    nengo.Connection(memory_noise, weighted_value[2], synapse=None)
    nengo.Connection(elapsed_time, task_monitor[1], synapse=None)
    nengo.Connection(emotional_state, arousal[1], synapse=None)
    # neural
    nengo.Connection(raw_value, weighted_value[:2])
    nengo.Connection(weighted_value, evidence, function=mult, synapse=tau, transform=tau)
    nengo.Connection(evidence, evidence, synapse=tau)
    nengo.Connection(evidence, utility)
    nengo.Connection(utility, decision.input)
    nengo.Connection(decision.output, motor_output.input)
    nengo.Connection(evidence, task_monitor[0], function=delta)
    nengo.Connection(task_monitor[1], arousal[0])
    nengo.Connection(arousal, utility, function=urgency)
    nengo.Connection(task_monitor, decision.stn.input, function=conflict)
    
    # Probes
    
    p_reward_memory = nengo.Probe(reward_memory)
    p_sensory_input = nengo.Probe(sensory_input)
    p_elapsed_time = nengo.Probe(elapsed_time)
    p_emotional_state = nengo.Probe(emotional_state)
    p_raw_value = nengo.Probe(raw_value)
    p_weighted_value = nengo.Probe(weighted_value)
    p_evidence = nengo.Probe(evidence)
    p_utility = nengo.Probe(utility)
    p_task_monitor = nengo.Probe(task_monitor)
    p_arousal = nengo.Probe(arousal)
    p_decision = nengo.Probe(decision.output)
    p_motor_output = nengo.Probe(motor_output.output)
    
    # Extra    
    decision.label = "decision (BG)"  # WTA action selection between A and B (past threshold)
    decision.strD1.label = 'STR_D1'
    decision.strD2.label = 'STR_D2'
    decision.stn.label = 'threshold (STN)'
    decision.gpe.label = 'GPE'
    decision.gpi.label = 'GPI'
    motor_output.label = "motor output"