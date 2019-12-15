import numpy as np
import nengo
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import nengolib
import hyperopt
import scipy
sns.set(style="white", context="poster")

''' Read empirical data from .csv files '''
cues = pd.read_csv('how_many_cues.csv', sep=';')
stim = pd.read_csv('input_stimuli.csv',sep=';')
choices = pd.read_csv('choices.csv', sep=';')

def read_values(trial):
    trdata = list(stim[['c1','c2','c3','c4','c5','c6','c1.1','c2.1','c3.1','c4.1','c5.1','c6.1']].loc[trial])
    return trdata[:len(trdata)//2], trdata[len(trdata)//2:]

def read_ncues_empirical(subj_id, trial):
    cols = list(cues.columns)
    return list(cues[cols[subj_id+1]])[trial]

def read_choice(subj_id, trial):
    cols = list(choices.columns)
    return list(choices[cols[subj_id+1]])[trial]

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

def is_correct_empirical(choice_empirical, trial):
    _, choice_correct = get_evidence_opt(trial)
    if choice_empirical == 'A' and choice_correct == 0:
        return True
    if choice_empirical == 'B' and choice_correct == 1:
        return True
    return False

def get_ncues_opt(trial):
    # the time at which the evidence that could accumulate for option 2 can no longer outweight the current evidence for option 1
    # (regardless of actual cue values on this trial)
    evidence_opt, choice_opt = get_evidence_opt(trial)
    for n in range(6):
        possible_opposing_evidence = 0
        if evidence_opt[n][choice_opt] > evidence_opt[n][choice_opt-1] + np.sum(validities[n+1:]):
            return n+1

def get_ncues_oracle(trial):
    # the time at which the evidence that will accumulate for option 2 can no longer outweight the current evidence for option 1
    # (given the actual cue values on this trial)
    evidence_opt, choice_opt = get_evidence_opt(trial)
    for n in range(6):
        if evidence_opt[n][choice_opt] > evidence_opt[-1][choice_opt-1]:
            return n+1

def get_ncues_greedy(trial):
    # the first time at which the evidence accumulated for one option outweights the evidence accumulated for the other option
    evidence_opt, choice_opt = get_evidence_opt(trial)
    for n in range(6):
        if evidence_opt[n][0] != evidence_opt[n][1]:
            return n+1


''' Model definition '''
validities = np.array([0.706, 0.688, 0.667, 0.647, 0.625, 0.6]) # validity of cues (from experiment)
T = 1.0  # time interval (s) for which the choices are presented

def run_model(trial, thr_int, thr_decay, seed=0):

    values_A, values_B = read_values(trial)
    values = [[values_A[n], values_B[n]] for n in range(len(values_A))]
    validity_process = nengo.processes.PresentInput(validities, presentation_time=T)
    value_process = nengo.processes.PresentInput(values, presentation_time=T)

    model = nengo.Network(seed=seed)
    with model:
        # Inputs
        validity_inpt = nengo.Node(validity_process)
        value_inpt = nengo.Node(value_process)
        threshold_inpt = nengo.Node(lambda t: t*thr_decay)
        validity_node = nengo.Ensemble(1, 3, neuron_type=nengo.Direct())
        evidence_node = nengo.Ensemble(1, 2, neuron_type=nengo.Direct())
        
        # Ensembles
        threshold = nengo.Ensemble(1000, 1, encoders=nengo.dists.Choice([[-1]]), intercepts=thr_int)  # dynamic threshold gates inputs to BG   
        multiply = nengo.Ensemble(2000, 3, radius=3)  # represents values dim=[0,1] and validities dim=[2]
        evidence = nengo.Ensemble(2000, 2, radius=4)  # 2D integrator accumulates weighted evidence
        gate = nengo.Ensemble(2000, 2, radius=4)  # relays information from evidence to decision
        decision = nengo.networks.BasalGanglia(n_neurons_per_ensemble=1000, dimensions=2)  # WTA action selection between A and B once threshold is reached

        # Connections
        nengo.Connection(value_inpt, multiply[0:2], synapse=None)
        nengo.Connection(validity_inpt, multiply[2], synapse=None)
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


def run_trials(n_trials=48, thr_decay=0, thr_int_low=-0.5, thr_int_high=-0.5, plot=True, seed=0):
    corrects_simulated = np.zeros((n_trials, 1))
    ncues_simulated = np.zeros((n_trials, 1))
    thr_int = nengo.dists.Uniform(thr_int_low, thr_int_high)
    for trial in range(n_trials):
        print('\ntrial: ', trial)
        data = run_model(trial, thr_int, thr_decay, seed)
        ncues_model = ncues(data['decision'])
        ncues_opt = get_ncues_opt(trial)
        ncues_greedy = get_ncues_greedy(trial)
        correct = is_correct(data['decision'], trial)
        corrects_simulated[trial] = is_correct(data['decision'], trial)
        ncues_simulated[trial] = ncues(data['decision'])
        print("n_cues model: ", int(ncues_simulated[trial][0]))
        print("correct model: ", corrects_simulated[trial][0]==1)
        if plot:
            make_plot(data, trial, ncues_model, ncues_opt, ncues_greedy, correct, thr_decay, thr_int)

    np.savez("plots5/data_thr_decay%.3f_thr_int%s.npz"%(thr_decay, thr_int),
        thr_decay=thr_decay, seed=seed, thr_int_low=thr_int_low, thr_int_high=thr_int_high,
        corrects_simulated=corrects_simulated, ncues_simulated=ncues_simulated)

    mean_model = 100*np.mean(corrects_simulated)
    fig, ax = plt.subplots(figsize=((12, 12)))
    ax.hist(ncues_simulated, bins=np.array([1,2,3,4,5,6,7])-0.5, rwidth=1, label='accuracy=%.2f%%'%mean_model)
    ax.set(xlabel='n_cues', ylabel='frequency', xticks=([1,2,3,4,5,6]),
        title="thr_decay=%.2f, thr_int=%s"%(thr_decay, thr_int))
    plt.legend()
    plt.savefig("plots5/ncues_distribution_thr_decay=%.2f, thr_int=%s.png"%(thr_decay, thr_int))

def make_plot(data, trial, ncues_model, ncues_opt, ncues_greedy, correct, thr_decay, thr_int):
    fig, (ax1, ax2) = plt.subplots(2, 1, sharex=True, figsize=((12, 12)))
    ax1.plot(data['times'], data['evidence'][:,0], label='evidence A', color='r')
    ax1.plot(data['times'], data['evidence'][:,1], label='evidence B', color='b')
    ax1.plot(data['times'], data['threshold'], label='threshold', color='k')
    ax1.plot(data['times'], data['gate'][:,0], label='gate A', linestyle='-.', color='r')
    ax1.plot(data['times'], data['gate'][:,1], label='gate B', linestyle='-.', color='b')
    ax1.plot(data['times'], data['evidence_node'][:,0], label='optimal A', color='r', linestyle="--")
    ax1.plot(data['times'], data['evidence_node'][:,1], label='optimal B', color='b', linestyle="--")
    ax2.plot(data['times'], data['decision'][:,0], color='r', label='A')
    ax2.plot(data['times'], data['decision'][:,1], color='b', label='B')
    ax2.axvline(x=ncues_model, alpha=0.5, label='ncues_model', color='r')
    ax2.axvline(x=ncues_opt, alpha=0.5, label='ncues_opt', color='b')
    ax2.axvline(x=ncues_greedy, alpha=0.5, label='ncues_greedy', color='m')
    ax1.set(xticks=(np.arange(1, 7)), ylim=((-1, 3)), ylabel='value',
        title='trial %s, correct=%s \n thr_decay=%.3f, thr_int=%s'%(trial, correct, thr_decay, thr_int))
    ax2.set(xticks=(np.arange(1, 7)), ylabel='BG values')
    # ax1.legend(loc='upper left')
    # ax2.legend(loc='upper left')
    plt.savefig("plots5/timeseries_thr_decay%.3f_thr_int%s_trial%s.png"%(thr_decay, thr_int, trial))
    plt.close()

def plot_participant_data():
    columns = ('trial', 'participant', 'correct', 'n_cues')
    df = pd.DataFrame(columns=columns)
    mean_ncues = np.zeros((14))
    mean_corrects = np.zeros((14))
    n_partitipants =  14
    n_trials = 48
    for participant in range(n_partitipants):
        corrects_participant = []
        n_cues_participant = []
        for trial in range(n_trials):
            correct = is_correct_empirical(read_choice(participant, trial), trial)
            n_cues = read_ncues_empirical(participant, trial)
            corrects_participant.append(correct)
            n_cues_participant.append(n_cues)
            df = df.append(pd.DataFrame([[participant, trial, correct, n_cues]], columns=columns), ignore_index=True)
        mean_ncues[participant] = np.mean(n_cues_participant)
        mean_corrects[participant] = np.mean(corrects_participant)

        fig, ax = plt.subplots(figsize=((12, 12)))
        ax.hist(n_cues_participant, bins=np.array([1,2,3,4,5,6,7])-0.5, rwidth=1, density=True, label='accuracy=%.2f%%'%(100*np.mean(corrects_participant)))
        ax.set(xlabel='n_cues', ylabel='frequency', xticks=([1,2,3,4,5,6]))
        plt.legend()
        plt.savefig("plots5/empirical_participant%s.png"%participant)

    fig, ax = plt.subplots(figsize=((12, 12)))
    ax.hist(df['n_cues'], bins=np.array([1,2,3,4,5,6,7])-0.5, rwidth=1, density=True,)
    ax.set(xlabel='n_cues', ylabel='frequency', title='all agents, all trials', xticks=([1,2,3,4,5,6]))
    plt.savefig("plots5/ncues_distplot.png")

    fig, ax = plt.subplots(figsize=((12, 12)))
    sns.scatterplot(mean_ncues, mean_corrects)
    ax.set(xlabel='mean n_cues', ylabel='mean accuracy', title='one point per agent', xlim=((1, 6)))
    plt.savefig("plots5/accuracy_vs_ncues_scatter.png")


# plot_participant_data()

run_trials(n_trials=2, thr_decay=0.15, thr_int_low=-1, thr_int_high=-0.8, seed=1)

