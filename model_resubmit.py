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

def run_model(values, weights, thr=3.0, urg=0.5, tau=0.1, n_neurons=1000, seed=0):

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

        # Probes
        p_weight_inpt = nengo.Probe(weight_inpt, synapse=None)
        p_value_inpt = nengo.Probe(value_inpt, synapse=None)
        p_value = nengo.Probe(value)
        p_hold = nengo.Probe(hold)
        p_evidence = nengo.Probe(evidence)
        p_gate = nengo.Probe(gate)
        p_decision = nengo.Probe(decision.output)
        p_motor = nengo.Probe(motor.output)

        # Ideal evidence accumulation (computed in math, compare to noisy integrator)
        val = nengo.Ensemble(1, 3, neuron_type=nengo.Direct())
        ev = nengo.Ensemble(1, 2, neuron_type=nengo.Direct(), label="ideal")
        nengo.Connection(value_inpt, val[:2], synapse=None)
        nengo.Connection(weight_inpt, val[2], synapse=None)
        nengo.Connection(val, ev, synapse=1/nengolib.signal.s, function=mult)
        p_ideal = nengo.Probe(ev, synapse=None)

    sim = nengo.Simulator(model, seed=seed, progress_bar=False)
    with sim:
        sim.run(6, progress_bar=True)

    return dict(
        times=sim.trange()[::10][10:],
        value=sim.data[p_value][10:],
        hold=sim.data[p_hold][10:],
        evidence=sim.data[p_evidence][10:],
        gate=sim.data[p_gate][10:],
        decision=sim.data[p_decision][10:],
        motor=sim.data[p_motor][10:],
        ideal=sim.data[p_ideal][10:],
        )

def read_values(trial):
    stim = pd.read_csv('input_stimuli.csv', sep=';')
    trdata = list(stim[['c1','c2','c3','c4','c5','c6','c1.1','c2.1','c3.1','c4.1','c5.1','c6.1']].loc[trial])
    A = trdata[:len(trdata)//2]
    B = trdata[len(trdata)//2:]
    return np.array([[A[n], B[n]] for n in range(len(A))])

def get_RT(data):
    RT_A = np.where(data[:,0] > 0.1)[0]
    RT_B = np.where(data[:,1] > 0.1)[0]
    if len(RT_A) > 0 and len(RT_B) == 0:
        return RT_A[0] * 0.01
    elif len(RT_B) > 0 and len(RT_A) == 0:
        return RT_B[0] * 0.01
    elif len(RT_A) > 0 and len(RT_B) > 0:
        return np.min([RT_A[0], RT_B[0]]) * 0.01
    else:
        raise

def get_choice(data):
    RT_A = np.where(data[:,0] > 0.1)[0]
    RT_B = np.where(data[:,1] > 0.1)[0]
    if len(RT_A) > 0 and len(RT_B) == 0:
        return 0  # 0 == A
    elif len(RT_B) > 0 and len(RT_A) == 0:
        return 1  # 1 == B
    elif len(RT_A) > 0 and len(RT_B) > 0:
        return np.argmin([RT_A[0], RT_B[0]])
    else:
        raise

def get_correct(values, weights):
    evidence_A = np.dot(values[:,0], weights)
    evidence_B = np.dot(values[:,1], weights)
    return np.argmax([evidence_A, evidence_B])

def make_plot(data, trial, RT, accuracy, choice, thr, urg):
    fig, ax = plt.subplots(1, 1, sharex=True, figsize=((12, 6)))
    # ax.plot(data['times'], data['value'][:,0], label="val A", color='r')
    # ax.plot(data['times'], data['value'][:,1], label="val B", color='b')
    ax.plot(data['times'], data['evidence'][:,0], label="evidence A", color='r')
    ax.plot(data['times'], data['evidence'][:,1], label="evidence B", color='b')
    ax.plot(data['times'], data['ideal'][:,0], label="ideal A", color='r', linestyle="--")
    ax.plot(data['times'], data['ideal'][:,1], label="ideal A", color='b', linestyle="--")
    # ax.plot(data['times'], data['gate'][:,0], label="gate A", color='r')
    # ax.plot(data['times'], data['gate'][:,1], label="gate B", color='b')
    # ax.plot(data['times'], data['decision'][:,0], label="dec A", linestyle=':', color='r')
    # ax.plot(data['times'], data['decision'][:,1], label="dec B", linestyle=':', color='b')
    ax.plot(data['times'], data['motor'][:,0], label="choose A", alpha=0.5, linestyle=":", color='r')
    ax.plot(data['times'], data['motor'][:,1], label="choose B", alpha=0.5, linestyle=":", color='b')
    ax.plot(data['times'], data['hold'], label='hold', color='k')
    ax.set(xlabel='time (s)', ylabel=r"$\hat{\mathbf{x}}$", ylim=((-0.25, 3)),
        xticks=[0,1,2,3,4,5,6], yticks=[0,1,2,3],
        title="thr=%.2f, urg=%.2f, trial=%s \n choice=%s (%s), RT=%.2fs"
        %(thr, urg, trial, 'A' if choice==0 else 'B', 'correct' if accuracy else 'incorrect', RT))
    ax.legend(loc='upper center', ncol=4, fancybox=True, shadow=True)
    plt.savefig("plots/thr%.2f_urg%.2f_trial%s.pdf"%(thr, urg, trial))
    plt.close()
    
def run_agent(n_trials=48, thr=3.0, urg=0.3, seed=0, plot=True):
    weights = np.array([0.706, 0.688, 0.667, 0.647, 0.625, 0.6]) # from noncompensatory experiment
    accuracies = np.zeros((n_trials, 1))
    RTs = np.zeros((n_trials, 1))
    for trial in range(n_trials):
        print('agent %s, trial %s: '%(seed, trial))
        values = read_values(trial)
        data = run_model(values, weights, thr, urg, seed)
        RT = get_RT(data['motor'])
        RTs[trial] = RT
        choice = get_choice(data['motor'])
        correct = get_correct(values, weights)
        accuracy = choice==correct
        accuracies[trial] = accuracy
        if plot:
            make_plot(data, trial, RT, accuracy, choice, thr, urg)
    np.savez("plots/data_thr%.2f_urg%.2f.npz"%(thr, urg), urg=urg, seed=seed, thr=thr, RTs=RTs, accuracies=accuracies)
    np.savez("data/%s"%seed, urg=urg, seed=seed, thr=thr, RTs=RTs, accuracies=accuracies) 

# Figures 3, 4, 5
# run_agent(n_trials=10, urg=0.4, thr=2.5, seed=2)
run_agent(n_trials=1, urg=0.3, thr=3.0, seed=3)
# run_agent(n_trials=10, urg=0.5, thr=2.0, seed=4)

# Figure 6
# run_agent(urg=0.5, thr=1.8, seed=107)
# run_agent(urg=0.29, thr=2.9, seed=110)

# Figure 7
# n_subjects = 14
# rng = np.random.RandomState(seed=1)
# urgs = np.linspace(0.3, 0.5, n_subjects)
# thrs = np.linspace(2.0, 3.0, n_subjects)
# rng.shuffle(urgs)
# rng.shuffle(thrs)
# for n in range(n_subjects):
#     run_agent(urg=urgs[n], thr=thrs[n], seed=n+15)
