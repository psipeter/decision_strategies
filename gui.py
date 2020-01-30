import numpy as np
import nengo
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import nengolib
import scipy
sns.set(style="white", context="poster")

''' Read empirical data from .csv files '''
cues = pd.read_csv('how_many_cues.csv', sep=';')
stim = pd.read_csv('input_stimuli.csv',sep=';')
choices = pd.read_csv('choices.csv', sep=';')
validities = np.array([0.706, 0.688, 0.667, 0.647, 0.625, 0.6]) # validity of cues (from experiment)
T = 1.0  # time interval (s) for which the choices are presented

def read_values(trial):
    trdata = list(stim[['c1','c2','c3','c4','c5','c6','c1.1','c2.1','c3.1','c4.1','c5.1','c6.1']].loc[trial])
    return trdata[:len(trdata)//2], trdata[len(trdata)//2:]

''' Process data from the simulation to determine when the model made a choice, and compare to the optimal time of choice '''
def get_evidence_opt(trial):
    values_A, values_B = read_values(trial)
    evidence_opt = np.zeros((6, 2))
    for n in range(6):
        evidence_opt[n] = np.array([values_A[n], values_B[n]]) * validities[n] + evidence_opt[n-1]
    choice_opt = np.argmax(evidence_opt[-1])
    return evidence_opt, choice_opt

def ncues(data, dt=0.001, thr=0.1):
    for ncues in range(6):
        data_range = data[int(T*ncues/dt):int(T*(ncues+1)/dt)]
        # if decision is A or B at the end of this time slot, model has decided
        if np.abs(data_range[-1, 0] - data_range[-1, 1]) > thr:
            return ncues+1
    return 6

def is_correct(data, trial, dt=0.001, thr=0.1):
    evidence_opt, choice_opt = get_evidence_opt(trial)
    for ncues in range(6):
        data_range = data[int(T*ncues/dt):int(T*(ncues+1)/dt)]
        # find which dimension of 'decision' is greater at the end of the time window, if beyond a difference threshold (same criteria as ncues)
        if np.abs(data_range[-1, 0] - data_range[-1, 1]) > thr:
            if data_range[-1, 0] > data_range[-1, 1]:
                # print('choose A after %s'%(ncues+1))
                return choice_opt == 0
            elif data_range[-1, 0] < data_range[-1, 1]:
                # print('choose B after %s'%(ncues+1))
                return choice_opt == 1
    best_end_utility = 0 if data_range[-1,0] > data_range[-1,1] else 1
    return best_end_utility == choice_opt


''' Model definition '''
trial=0
Tint=3.0
Tdecay=0.4
seed=0
noise=5e-2

values_A, values_B = read_values(trial)
values = [[values_A[n], values_B[n]] for n in range(len(values_A))]
validity_process = nengo.processes.PresentInput(validities, presentation_time=T)
value_process = nengo.processes.PresentInput(values, presentation_time=T)

model = nengo.Network(seed=seed)
with model:
    # Inputs
    validity_inpt = nengo.Node(validity_process, label="validity")
    value_inpt = nengo.Node(value_process, label="value")
    threshold_inpt = nengo.Node(lambda t: t*Tdecay + 10*(t>5.5), label="threshold")
    noise_inpt = nengo.Node(nengo.processes.WhiteSignal(period=6, high=10, rms=noise, seed=seed), label="noise")
    # validity_node = nengo.Ensemble(1, 3, neuron_type=nengo.Direct())  # compute in math
    # evidence_node = nengo.Ensemble(1, 2, neuron_type=nengo.Direct())  # compute in math
    
    # Ensembles
    threshold = nengo.Ensemble(1000, 1,  # dynamic threshold gates inputs to BG
        encoders=nengo.dists.Choice([[-1]]), intercepts=nengo.dists.Choice([-Tint]), label="lPFC (threshold)")  
    multiply = nengo.Ensemble(2000, 3, radius=3, label="OFC (multiply)")  # represents values dim=[0,1] and validities dim=[2]
    evidence = nengo.Ensemble(2000, 2, radius=4, label="dlPFC (evidence)")  # 2D integrator accumulates weighted evidence
    gate = nengo.Ensemble(2000, 2, radius=4, label="pSMA (gate)")  # relays information from evidence to decision
    decision = nengo.networks.BasalGanglia(n_neurons_per_ensemble=1000, dimensions=2)  # WTA action selection between A and B once threshold is reached
    decision.label = "BG (decision)"
    decision.strD1.label = 'STR_D1'
    decision.strD2.label = 'STR_D2'
    decision.stn.label = 'STN'
    decision.gpe.label = 'GPE'
    decision.gpi.label = 'GPI'

    # Connections
    nengo.Connection(value_inpt, multiply[0:2], synapse=None)  # visual inputs
    nengo.Connection(validity_inpt, multiply[2], synapse=None)  # external memory of validities
    nengo.Connection(noise_inpt, multiply[2], synapse=None)  # noisy memory recall of validities
    # accumulate evidence for choice A and B by feeding weighted values into a 2D integrator
    # function multiplies input validies by input values
    # nengo.Connection(multiply, evidence[0], synapse=0.1, function=lambda x: x[0]*x[2], transform=0.1)
    # nengo.Connection(multiply, evidence[1], synapse=0.1, function=lambda x: x[1]*x[2], transform=0.1)
    nengo.Connection(multiply, evidence, synapse=0.1, function=lambda x: [x[0]*x[2], x[1]*x[2]], transform=0.1)
    nengo.Connection(evidence, evidence, synapse=0.1)
    # pass this evidence through a gate to the BG
    nengo.Connection(evidence, gate, synapse=0.1)
    nengo.Connection(gate, decision.input, synapse=0.1)
    # close the gate unless the difference in accumulated evidence for A and B exceeds threshold
    nengo.Connection(evidence, threshold, function=lambda x: np.abs(x[0]-x[1]), synapse=0.01)
    # open the gate near the end of the task
    nengo.Connection(threshold_inpt, threshold, synapse=None)
    # do this via inhibition from threshold to gate, active by default
    nengo.Connection(threshold, gate.neurons, transform=-1e3*np.ones((gate.n_neurons, 1)), synapse=0.01)
    # ideal evidence accumulation (computed in math, compare to noisy integrator)
    # nengo.Connection(value_inpt, validity_node[0:2])
    # nengo.Connection(validity_inpt, validity_node[2])
    # nengo.Connection(validity_node, evidence_node, synapse=1/nengolib.signal.s, function=lambda x: [x[0]*x[2], x[1]*x[2]])

    # Probes
    p_validity_inpt = nengo.Probe(validity_inpt, synapse=None)
    p_value_inpt = nengo.Probe(value_inpt, synapse=None)
    p_threshold = nengo.Probe(threshold, synapse=0.1)
    p_multiply = nengo.Probe(multiply, synapse=0.1)
    p_evidence = nengo.Probe(evidence, synapse=0.1)
    p_gate = nengo.Probe(gate, synapse=0.1)
    p_decision = nengo.Probe(decision.output, synapse=0.1)
    # p_evidence_node = nengo.Probe(evidence_node, synapse=0.1)