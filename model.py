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
def run_model(trial, Tint, Tdecay, seed=0, noise=5e-2):

    values_A, values_B = read_values(trial)
    values = [[values_A[n], values_B[n]] for n in range(len(values_A))]
    validity_process = nengo.processes.PresentInput(validities, presentation_time=T)
    value_process = nengo.processes.PresentInput(values, presentation_time=T)

    model = nengo.Network(seed=seed)
    with model:
        # Inputs
        validity_inpt = nengo.Node(validity_process)
        value_inpt = nengo.Node(value_process)
        threshold_inpt = nengo.Node(lambda t: t*Tdecay + 10*(t>5.5))
        noise_inpt = nengo.Node(nengo.processes.WhiteSignal(period=6, high=10, rms=noise, seed=seed))
        validity_node = nengo.Ensemble(1, 3, neuron_type=nengo.Direct())  # compute in math
        evidence_node = nengo.Ensemble(1, 2, neuron_type=nengo.Direct())  # compute in math
        
        # Ensembles
        threshold = nengo.Ensemble(1000, 1, encoders=nengo.dists.Choice([[-1]]), intercepts=nengo.dists.Choice([-Tint])) # dynamic threshold gates inputs to BG
        multiply = nengo.Ensemble(2000, 3, radius=3)  # represents values dim=[0,1] and validities dim=[2]
        evidence = nengo.Ensemble(2000, 2, radius=4)  # 2D integrator accumulates weighted evidence
        gate = nengo.Ensemble(2000, 2, radius=4)  # relays information from evidence to decision
        decision = nengo.networks.BasalGanglia(n_neurons_per_ensemble=1000, dimensions=2)  # WTA action selection between A and B once threshold is reached

        # Connections
        nengo.Connection(value_inpt, multiply[0:2], synapse=None)  # visual inputs
        nengo.Connection(validity_inpt, multiply[2], synapse=None)  # external memory of validities
        nengo.Connection(noise_inpt, multiply[2], synapse=None)  # noisy memory recall of validities
        # accumulate evidence for choice A and B by feeding weighted values into a 2D integrator
        # function multiplies input validies by input values
        nengo.Connection(multiply, evidence[0], synapse=0.1, function=lambda x: x[0]*x[2], transform=0.1)
        nengo.Connection(multiply, evidence[1], synapse=0.1, function=lambda x: x[1]*x[2], transform=0.1)
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
        nengo.Connection(value_inpt, validity_node[0:2])
        nengo.Connection(validity_inpt, validity_node[2])
        nengo.Connection(validity_node, evidence_node, synapse=1/nengolib.signal.s, function=lambda x: [x[0]*x[2], x[1]*x[2]])

        # Probes
        p_validity_inpt = nengo.Probe(validity_inpt, synapse=None)
        p_value_inpt = nengo.Probe(value_inpt, synapse=None)
        p_threshold = nengo.Probe(threshold, synapse=0.1)
        p_multiply = nengo.Probe(multiply, synapse=0.1)
        p_evidence = nengo.Probe(evidence, synapse=0.1)
        p_gate = nengo.Probe(gate, synapse=0.1)
        p_decision = nengo.Probe(decision.output, synapse=0.1)
        p_evidence_node = nengo.Probe(evidence_node, synapse=0.1)

    sim = nengo.Simulator(model, seed=seed, progress_bar=True)
    with sim:
        sim.run(6, progress_bar=True)

    return dict(
        times=sim.trange(),
        threshold=sim.data[p_threshold],
        multiply=sim.data[p_multiply],
        evidence=sim.data[p_evidence],
        gate=sim.data[p_gate],
        decision=sim.data[p_decision],
        evidence_node=sim.data[p_evidence_node],
        )


def run_agent(n_trials=48, Tdecay=0.3, Tint=3.0, plot=True, seed=0, p=0):
    corrects_simulated = np.zeros((n_trials, 1))
    ncues_simulated = np.zeros((n_trials, 1))
    for trial in range(n_trials):
        print('\ntrial: ', trial)
        data = run_model(trial, Tint, Tdecay, seed)
        ncues_model = ncues(data['decision'])
        correct = is_correct(data['decision'], trial)
        corrects_simulated[trial] = is_correct(data['decision'], trial)
        ncues_simulated[trial] = ncues(data['decision'])
        print("n_cues model: ", int(ncues_simulated[trial][0]))
        print("correct model: ", corrects_simulated[trial][0]==1)
        if plot:
            make_plot(data, trial, ncues_model, correct, Tdecay, Tint)

    np.savez("plots/data_Tdecay%.2f_Tint%.2f.npz"%(Tdecay, Tint),
        Tdecay=Tdecay, seed=seed, Tint=Tint,
        corrects_simulated=corrects_simulated, ncues_simulated=ncues_simulated)
    np.savez("data/%s"%p,
        Tdecay=Tdecay, seed=seed, Tint=Tint,
        corrects_simulated=corrects_simulated, ncues_simulated=ncues_simulated)    

    mean_model = 100*np.mean(corrects_simulated)
    fig, ax = plt.subplots(figsize=((12, 12)))
    ax.hist(ncues_simulated, bins=np.array([1,2,3,4,5,6,7])-0.5, rwidth=1, density=True, label='accuracy=%.2f%%'%mean_model)
    ax.set(xlabel='n_cues', ylabel='frequency', xticks=([1,2,3,4,5,6]),
        title="Tdecay=%.2f, Tint=%.2f"%(Tdecay, Tint))
    plt.legend()
    plt.savefig("plots/ncues_distribution_Tdecay%.2f, thr%.2f.png"%(Tdecay, Tint))

def make_plot(data, trial, ncues_model, correct, Tdecay, Tint):
    sns.set(style="white", context="poster")
    fig, ax = plt.subplots(1, 1, sharex=True, figsize=((12, 6)))
    ax.plot(data['times'], data['evidence'][:,0], label=r"$E_A$", color='r')
    ax.plot(data['times'], data['evidence'][:,1], label=r"$E_B$", color='b')
    ax.plot(data['times'], data['threshold'], label='rIFC', color='k')
    ax.plot(data['times'], data['gate'][:,0], linestyle='-.', color='r')
    ax.plot(data['times'], data['gate'][:,1], linestyle='-.', color='b')
    ax.plot(data['times'], data['evidence_node'][:,0], color='r', linestyle="--")
    ax.plot(data['times'], data['evidence_node'][:,1], color='b', linestyle="--")
    ax.set(xticks=(np.arange(1, 7)), yticks=([0, 1, 2, 3]), ylim=((-0.25, 3)), ylabel=r"$\hat{\mathbf{x}}$", xlabel='time (cues)')
    ax.legend(loc='upper left')
    plt.tight_layout()
    plt.savefig("plots/Tdecay%.2f_Tint%.2f_trial%s.png"%(Tdecay, Tint, trial))
    plt.close()

def make_fig3():
    data_top = run_model(trial=1, Tint=3.0, Tdecay=0.4, seed=0)
    data_middle = run_model(trial=0, Tint=3.0, Tdecay=0.4, seed=0)
    data_bottom = run_model(trial=10, Tint=2.0, Tdecay=0.5, seed=0)

    sns.set(style="white", context="poster")
    fig, (ax1, ax2, ax3) = plt.subplots(3, 1, sharex=True, figsize=((8, 12)))
    ax1.plot(data_top['times'], data_top['evidence'][:,0], label=r"$E_A$", color='r')
    ax1.plot(data_top['times'], data_top['evidence'][:,1], label=r"$E_B$", color='b')
    ax1.plot(data_top['times'], data_top['threshold'], label='rIFC', color='k')
    ax1.plot(data_top['times'], data_top['gate'][:,0], linestyle='-.', color='r')
    ax1.plot(data_top['times'], data_top['gate'][:,1], linestyle='-.', color='b')
    ax1.plot(data_top['times'], data_top['evidence_node'][:,0], color='r', linestyle="--")
    ax1.plot(data_top['times'], data_top['evidence_node'][:,1], color='b', linestyle="--")

    ax2.plot(data_middle['times'], data_middle['evidence'][:,0], color='r')
    ax2.plot(data_middle['times'], data_middle['evidence'][:,1], color='b')
    ax2.plot(data_middle['times'], data_middle['threshold'], color='k')
    ax2.plot(data_middle['times'], data_middle['gate'][:,0], linestyle='-.', color='r')
    ax2.plot(data_middle['times'], data_middle['gate'][:,1], linestyle='-.', color='b')
    ax2.plot(data_middle['times'], data_middle['evidence_node'][:,0], color='r', linestyle="--")
    ax2.plot(data_middle['times'], data_middle['evidence_node'][:,1], color='b', linestyle="--")

    ax3.plot(data_bottom['times'], data_bottom['evidence'][:,0], color='r')
    ax3.plot(data_bottom['times'], data_bottom['evidence'][:,1], color='b')
    ax3.plot(data_bottom['times'], data_bottom['threshold'], color='k')
    ax3.plot(data_bottom['times'], data_bottom['gate'][:,0], linestyle='-.', color='r')
    ax3.plot(data_bottom['times'], data_bottom['gate'][:,1], linestyle='-.', color='b')
    ax3.plot(data_bottom['times'], data_bottom['evidence_node'][:,0], color='r', linestyle="--")
    ax3.plot(data_bottom['times'], data_bottom['evidence_node'][:,1], color='b', linestyle="--")

    ax1.legend(loc='upper center', bbox_to_anchor=(0.5, 1.3), ncol=3, fancybox=True, shadow=True)
    ax1.set(yticks=([0, 1, 2, 3]), ylim=((-0.25, 3)), ylabel=r"$\hat{\mathbf{x}}$")
    ax2.set(yticks=([0, 1, 2, 3]), ylim=((-0.25, 3)), ylabel=r"$\hat{\mathbf{x}}$")
    ax3.set(xticks=(np.arange(1, 7)), yticks=([0, 1, 2, 3]), ylim=((-0.25, 3)), ylabel=r"$\hat{\mathbf{x}}$", xlabel='time (cues)')
    plt.tight_layout()
    plt.savefig("plots/three_timeseries.png")
    plt.close()

# make_fig3()

run_agent(n_trials=10, Tdecay=0.5, Tint=2.0, seed=0, p=0)

# n_subjects = 30
# rng = np.random.RandomState(seed=0)
# for n in range(n_subjects):
#     run_agent(Tdecay=rng.uniform(0.3, 0.5), Tint=rng.uniform(1.5, 3.0), seed=n, p=n)