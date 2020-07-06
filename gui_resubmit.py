import numpy as np
import nengo
from nengo.dists import Choice
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import nengolib
import scipy
# sns.set(style="white", context="paper")
sns.set(style="white", context="poster")

# Read stimulus data for a given trial
def read_values(trial):
    stim = pd.read_csv('input_stimuli.csv', sep=';')
    trdata = list(stim[['c1','c2','c3','c4','c5','c6','c1.1','c2.1','c3.1','c4.1','c5.1','c6.1']].loc[trial])
    A = trdata[:len(trdata)//2]
    B = trdata[len(trdata)//2:]
    return np.array([[A[n], B[n]] for n in range(len(A))])

# Parameters (variable)
trial = 3
urg = 0.5
thr = 2.0
seed = 4
# Figure 3: trial=6, urg=0.4, thr=2.5, seed=2)
# Figure 4: trial=0, urg=0.3, thr=3.0, seed=3)
# Figure 5: trial=3, urg=0.5, thr=2.0, seed=4)

# Parameters (fixed)
values = read_values(trial)
weights = np.array([0.706, 0.688, 0.667, 0.647, 0.625, 0.6])
tau = 0.1
n_neurons = 1000

value_process = nengo.processes.PresentInput(values, presentation_time=1.0)
weight_process = nengo.processes.PresentInput(weights, presentation_time=1.0)
noise_process = nengo.processes.WhiteSignal(period=6, high=10, rms=6e-2, seed=seed)
urgency_process = lambda t: thr - t*urg - thr*(t>5.8)

mult = lambda x: [x[0]*x[2], x[1]*x[2]]  # multiply weights by values
delta = lambda x: -np.abs(x[0]-x[1])  # negative of difference in accumulated evidence
inhib = -1e3*np.ones((n_neurons, 1))

model = nengo.Network(seed=seed)
with model:
    # Parameters
    model.config[nengo.Connection].synapse = 0.01
    model.config[nengo.Probe].synapse = 0.01
    model.config[nengo.Probe].sample_every = 0.01

    # Inputs
    value_inpt = nengo.Node(value_process)
    weight_inpt = nengo.Node(weight_process)
    urgency_inpt = nengo.Node(urgency_process)
    noise_inpt = nengo.Node(noise_process)

    # Ensembles
    value = nengo.Ensemble(n_neurons, 3, radius=3)  # represents values dim=[0,1] and weights dim=[2]
    evidence = nengo.Ensemble(n_neurons, 2, radius=4)  # 2D integrator accumulates weighted evidence
    hold = nengo.Ensemble(n_neurons, 1, encoders=Choice([[1]]), intercepts=Choice([0])) # monitor uncertainty+urgency, hold signal through gate
    gate = nengo.Ensemble(n_neurons, 2, radius=4)  # relays information from evidence to decision
    decision = nengo.networks.BasalGanglia(2, n_neurons)  # WTA action selection between A and B once threshold is reached
    motor = nengo.networks.Thalamus(2, n_neurons, threshold=0.6)  # mutual inhibition between BG outputs for motor execution

    # Connections
    nengo.Connection(value_inpt, value[0:2], synapse=None)  # sensory inputs
    nengo.Connection(weight_inpt, value[2], synapse=None)  # memory of weights
    nengo.Connection(noise_inpt, value[2], synapse=None)  # noisy recall of weights
    nengo.Connection(urgency_inpt, hold, synapse=None)  # ramping urgency signal
    nengo.Connection(value, evidence, function=mult, synapse=tau, transform=tau)  # multiply values and weights, feed to integrator
    nengo.Connection(evidence, evidence, synapse=tau)  # recurrent connection for evidence buffer
    nengo.Connection(evidence, gate)
    nengo.Connection(gate, decision.input, transform=0.4)
    nengo.Connection(evidence, hold, function=delta)  # calculate certainty
    nengo.Connection(hold, gate.neurons, transform=inhib)  # inhibitory control of gate
    nengo.Connection(decision.output, motor.input)  # final action selection


    # Labels
    value_inpt.label = "value input"
    weight_inpt.label = "weight memory"
    noise_inpt.label = "noise"
    urgency_inpt.label = "elapsed time"
    value.label = "value (OFC)"
    evidence.label = "evidence (dlPFC)"
    gate.label = "gate (pSMA)"
    hold.label = "hold (rIFC)"
    decision.label = "winner-take-all (BG)"  # WTA action selection between A and B (past threshold)
    decision.strD1.label = 'STR_D1'
    decision.strD2.label = 'STR_D2'
    decision.stn.label = 'threshold (STN)'
    decision.gpe.label = 'GPE'
    decision.gpi.label = 'GPI'
    motor.label = "motor output"
    ideal_evidence = nengo.Network(label="ideal")
    ideal_evidence.label = "ideal evidence"
    with ideal_evidence:
        # Ideal evidence accumulation (computed in math, compare to noisy integrator)
        val = nengo.Ensemble(1, 3, neuron_type=nengo.Direct())
        ev = nengo.Ensemble(1, 2, neuron_type=nengo.Direct())
        nengo.Connection(value_inpt, val[:2], synapse=None)
        nengo.Connection(weight_inpt, val[2], synapse=None)
        nengo.Connection(val, ev, synapse=1/nengolib.signal.s, function=mult)
