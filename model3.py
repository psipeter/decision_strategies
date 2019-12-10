import numpy as np
import nengo
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import nengolib
import hyperopt
import scipy
sns.set(style="white", context="talk")

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

def run_model(trial, gain_in, threshold_in):

    values_A, values_B = read_values(trial)
    values = [[values_A[n], values_B[n]] for n in range(len(values_A))]
    validity_process = nengo.processes.PresentInput(validities, presentation_time=T)
    value_process = nengo.processes.PresentInput(values, presentation_time=T)

    model = nengo.Network(seed=0)
    with model:
        # Inputs
        validity_inpt = nengo.Node(validity_process)
        value_inpt = nengo.Node(value_process)
        gain_inpt = nengo.Node(gain_in)
        threshold_inpt = nengo.Node(lambda t: t > 5.5)
        validity_node = nengo.Ensemble(1, 3, neuron_type=nengo.Direct())
        evidence_node = nengo.Ensemble(1, 2, neuron_type=nengo.Direct())
        
        # Ensembles
        gain = nengo.Ensemble(200, 1)  # represents emotional modulation of validities
        threshold = nengo.Ensemble(200, 1, encoders=nengo.dists.Choice([[-1]]), intercepts=nengo.dists.Choice([-threshold_in]))  # gates inputs to BG   
        multiply = nengo.Ensemble(2000, 3, radius=4)  # represents values dim=[0,1] and validities dim=[2]
        evidence = nengo.Ensemble(2000, 2, radius=4)  # 2D integrator accumulates weighted evidence
        gate = nengo.Ensemble(400, 2, radius=4)  # relays information from evidence to decision
        decision = nengo.networks.BasalGanglia(n_neurons_per_ensemble=200, dimensions=2)  # WTA action selection between A and B once threshold is reached

        # Connections
        nengo.Connection(gain_inpt, gain, synapse=None)
        nengo.Connection(value_inpt, multiply[0:2], synapse=None)
        nengo.Connection(validity_inpt, multiply[2], synapse=None)
        nengo.Connection(gain, multiply[2], synapse=0.1)
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
        nengo.Connection(threshold_inpt, threshold, synapse=None, transform=10)
        # do this via inhibition from threshold to gate, active by default
        nengo.Connection(threshold, gate.neurons, transform=-1e3*np.ones((gate.n_neurons, 1)), synapse=0.01)
        # ideal evidence accumulation (computed in math, compare to noisy integrator)
        nengo.Connection(value_inpt, validity_node[0:2])
        nengo.Connection(validity_inpt, validity_node[2])
        nengo.Connection(validity_node, evidence_node,
            synapse=1/nengolib.signal.s, function=lambda x: [x[0]*x[2], x[1]*x[2]])

        # Probes
        p_validity_inpt = nengo.Probe(validity_inpt, synapse=None)
        p_value_inpt = nengo.Probe(value_inpt, synapse=None)
        p_gain = nengo.Probe(gain, synapse=0.1)
        p_threshold = nengo.Probe(threshold, synapse=0.1)
        p_multiply = nengo.Probe(multiply, synapse=0.1)
        p_evidence = nengo.Probe(evidence, synapse=0.1)
        p_gate = nengo.Probe(gate, synapse=0.1)
        p_decision = nengo.Probe(decision.output, synapse=0.1)
        p_evidence_node = nengo.Probe(evidence_node, synapse=0.1)

    sim = nengo.Simulator(model, seed=0, progress_bar=True)
    with sim:
        sim.run(6, progress_bar=True)

    return dict(
        times=sim.trange(),
        threshold=sim.data[p_threshold],
        multiply=sim.data[p_multiply],
        evidence=sim.data[p_evidence],
        gain=sim.data[p_gain],
        gate=sim.data[p_gate],
        decision=sim.data[p_decision],
        evidence_node=sim.data[p_evidence_node],
        )

''' Experimental trials '''

def run_trials(
    participant=0,
    n_trials_train=48,
    n_trials_test=48,
    delta_gain=1e-1,
    delta_threshold=2e-1,
    strategy=1,  # 0 favors speed, 1 favors accuracy
    bias=0,  # -1 favors accuracy by cautious inspection, 1 favors speed by attention-narrowing
    rng=np.random.RandomState(seed=0),
    plot=False):

    # Training
    gain = 0
    threshold = 0.5
    trial_list = np.arange(48)
    rng.shuffle(trial_list)
    for trial in range(n_trials_train):
        trial = trial_list[np.mod(trial, 48)]
        if np.mod(trial, 48) == 0:
            rng.shuffle(trial_list)
        print('training trial: ', trial)
        data = run_model(trial, gain, threshold)

        # determine the timing and correctness of model behavior
        ncues_model = ncues(data['decision'])
        ncues_opt = get_ncues_opt(trial)
        ncues_greedy = get_ncues_greedy(trial)
        correct = is_correct(data['decision'], trial)
        print('correct:', correct==1)
        print('ncues_model:', ncues_model)
        print('ncues_opt', ncues_opt)
        print('ncues_greedy', ncues_greedy)
        if plot:
            make_plot(data, trial, ncues_model, ncues_opt, ncues_greedy, correct, gain, threshold, 'train', participant)
        # update gain and threshold for next trial
        reward_accuracy = -(not correct) * strategy  # -1 if incorrect, 0 if correct, scaled by strategy
        reward_speed = - (ncues_model-1) * (1-strategy) / 5  # -1 if chose at ncues=6, 0 if chose at ncues=1, scaled by inverse strategy
        gain += delta_gain * -reward_speed
        threshold += delta_threshold * (reward_speed - reward_accuracy)
        np.savez("data_train.npz", gain=gain, threshold=threshold, strategy=strategy, participant=participant,
            delta_gain=delta_gain, delta_threshold=delta_threshold)

    # Test under learned strategy
    print("final: gain=%.3f, threshold=%.3f"%(gain, threshold))
    corrects_simulated = np.zeros((n_trials_test, 1))
    corrects_empirical = np.zeros((n_trials_test, 1))
    ncues_simulated = np.zeros((n_trials_test, 1))
    ncues_empirical = np.zeros((n_trials_test, 1))
    for trial in range(n_trials_test):
        print('testing trial: ', trial)
        data = run_model(trial, gain, threshold)

        ncues_model = ncues(data['decision'])
        ncues_opt = get_ncues_opt(trial)
        ncues_greedy = get_ncues_greedy(trial)
        correct = is_correct(data['decision'], trial)
        corrects_simulated[trial] = is_correct(data['decision'], trial)
        ncues_simulated[trial] = ncues(data['decision'])
        corrects_empirical[trial] = is_correct_empirical(read_choice(participant, trial), trial)
        ncues_empirical[trial] = read_ncues_empirical(participant, trial)
        print("n_cues model: ", ncues_simulated[trial])
        print("n_cues empirical: ", ncues_empirical[trial])
        print("correct model: ", corrects_simulated[trial])
        print("correct empirical: ", corrects_empirical[trial])
        if plot:
            make_plot(data, trial, ncues_model, ncues_opt, ncues_greedy, correct, gain, threshold, 'test', participant)

    np.savez("data_test.npz", gain=gain, threshold=threshold, strategy=strategy, participant=participant,
        delta_gain=delta_gain, delta_threshold=delta_threshold, corrects_simulated=corrects_simulated, ncues_simulated=ncues_simulated, corrects_empirical=corrects_empirical, ncues_empirical=ncues_empirical)

    mean_model = 100*np.mean(corrects_simulated)
    mean_empirical = 100*np.mean(corrects_empirical)
    loss = scipy.stats.entropy(ncues_simulated, ncues_empirical)
    # if plot:
    fig, ax = plt.subplots(figsize=((12, 12)))
    sns.distplot(ncues_simulated, ax=ax, label='model, accuracy=%.1f%%'%mean_model, bins=[1,2,3,4,5,6], kde=False)
    sns.distplot(ncues_empirical, ax=ax, label='empirical, accuracy=%.1f%%'%mean_empirical, bins=[1,2,3,4,5,6], kde=False)
    ax.set(xlabel='number of requested cues before decision', ylabel='frequency', title="entropy=%.5f"%loss, xlim=((0, 6)))
    plt.legend()
    plt.savefig("plots3/%s/ncues_distribution.png"%participant)


    if bias != 0:
        # Test under learned stragey + bias
        print("biasing gain=%.3f, threshold=%.3f with bias=%.3f"%(gain, threshold, bias))
        corrects_simulated = np.zeros((n_trials_test, 1))
        corrects_empirical = np.zeros((n_trials_test, 1))
        ncues_simulated = np.zeros((n_trials_test, 1))
        ncues_empirical = np.zeros((n_trials_test, 1))
        for trial in range(n_trials_test):
            print('testing trial: ', trial)
            data = run_model(trial, gain+bias, threshold)  # bias modulates gain, leading to stronger/weaker evidence accumulation
            ncues_model = ncues(data['decision'])
            ncues_opt = get_ncues_opt(trial)
            ncues_greedy = get_ncues_greedy(trial)
            correct = is_correct(data['decision'], trial)
            corrects_simulated[trial] = is_correct(data['decision'], trial)
            ncues_simulated[trial] = ncues(data['decision'])
            corrects_empirical[trial] = is_correct_empirical(read_choice(participant, trial), trial)
            ncues_empirical[trial] = read_ncues_empirical(participant, trial)
            print("n_cues model: ", ncues_simulated[trial])
            print("n_cues empirical: ", ncues_empirical[trial])
            print("correct model: ", corrects_simulated[trial])
            print("correct empirical: ", corrects_empirical[trial])
            if plot:
                make_plot(data, trial, ncues_model, ncues_opt, ncues_greedy, correct, gain+bias, threshold, 'test_bias', participant)

        np.savez("data_test_bias.npz", gain=gain, threshold=threshold, strategy=strategy, participant=participant, delta_gain=delta_gain, delta_threshold=delta_threshold, corrects_simulated=corrects_simulated, ncues_simulated=ncues_simulated, corrects_empirical=corrects_empirical, ncues_empirical=ncues_empirical, bias=bias)

        mean_model = 100*np.mean(corrects_simulated)
        mean_empirical = 100*np.mean(corrects_empirical)
        loss = scipy.stats.entropy(ncues_simulated, ncues_empirical)
        # if plot:
        fig, ax = plt.subplots(figsize=((12, 12)))
        sns.distplot(ncues_simulated, ax=ax, label='model, accuracy=%.1f%%'%mean_model, bins=[1,2,3,4,5,6], kde=False)
        sns.distplot(ncues_empirical, ax=ax, label='empirical, accuracy=%.1f%%'%mean_empirical, bins=[1,2,3,4,5,6], kde=False)
        ax.set(xlabel='number of requested cues before decision', ylabel='frequency', title="entropy=%.5f"%loss, xlim=((0, 6)))
        plt.legend()
        plt.savefig("plots3/%s/ncues_distribution_biased.png"%participant)


    return gain, threshold

def make_plot(data, trial, ncues_model, ncues_opt, ncues_greedy, correct, gain, threshold, phase, participant):
    fig, (ax1, ax2) = plt.subplots(2, 1, sharex=True, figsize=((12, 12)))
    ax1.plot(data['times'], data['evidence'][:,0], label='evidence A', color='r')
    ax1.plot(data['times'], data['evidence'][:,1], label='evidence B', color='b')
    ax1.plot(data['times'], data['threshold'], label='threshold (ens)', color='k')
    ax1.axhline(y=threshold, alpha=0.5, label='threshold (target)', color='k')
    ax1.plot(data['times'], data['gate'][:,0], label='gate A', linestyle='-.', color='r')
    ax1.plot(data['times'], data['gate'][:,1], label='gate B', linestyle='-.', color='b')
    ax1.plot(data['times'], data['evidence_node'][:,0], label='optimal A', color='r', linestyle="--")
    ax1.plot(data['times'], data['evidence_node'][:,1], label='optimal B', color='b', linestyle="--")
    ax2.plot(data['times'], data['decision'][:,0], color='r', label='A')
    ax2.plot(data['times'], data['decision'][:,1], color='b', label='B')
    ax2.axvline(x=ncues_model, alpha=0.5, label='ncues_model', color='r')
    ax2.axvline(x=ncues_opt, alpha=0.5, label='ncues_opt', color='b')
    ax2.axvline(x=ncues_greedy, alpha=0.5, label='ncues_greedy', color='m')
    ax1.set(xticks=([0, 1, 2, 3, 4, 5, 6]), ylim=((-1, 3)), ylabel='value',
        title='trial %s, gain=%.3f, threshold(target)=%.3f, correct=%s'%(trial, gain, threshold, correct))
    ax2.set(xticks=([0, 1, 2, 3, 4, 5, 6]), ylabel='BG values')
    ax1.legend(loc='upper left')
    ax2.legend(loc='upper left')
    plt.savefig("plots3/%s/timeseries_%s_%s.png"%(participant, phase, trial))
    plt.close()

def plot_participant_data():
    columns = ('trial', 'participant', 'correct', 'n_cues')
    df = pd.DataFrame(columns=columns)
    mean_ncues = np.zeros((14))
    mean_corrects = np.zeros((14))
    n_partitipants =  14
    n_trials_test = 48
    for participant in range(n_partitipants):
        corrects_participant = []
        n_cues_participant = []
        for trial in range(n_trials_test):
            correct = is_correct_empirical(read_choice(participant, trial), trial)
            n_cues = read_ncues_empirical(participant, trial)
            corrects_participant.append(correct)
            n_cues_participant.append(n_cues)
            df = df.append(pd.DataFrame([[participant, trial, correct, n_cues]], columns=columns), ignore_index=True)
        mean_ncues[participant] = np.mean(n_cues_participant)
        mean_corrects[participant] = np.mean(corrects_participant)

        fig, ax = plt.subplots(figsize=((12, 12)))
        sns.distplot(n_cues_participant, ax=ax, bins=[1,2,3,4,5,6], kde=False, # rwidth=1, align='center',
            label='empirical, accuracy=%.2f%%'%np.mean(corrects_participant))
        ax.set(xlabel='number of requested cues before decision', ylabel='frequency', xlim=((0, 6)))
        plt.legend()
        plt.savefig("plots3/%s/empirical.png"%participant)

    fig, ax = plt.subplots(figsize=((12, 12)))
    sns.barplot('correct', 'n_cues', data=df)
    plt.savefig("plots3/ncues_vs_correct_barplot.png")

    fig, ax = plt.subplots(figsize=((12, 12)))
    sns.distplot(df['n_cues'], bins=np.arange(7), kde=False)
    ax.set(ylabel='frequency', xticks=(np.arange(1, 7)))
    plt.savefig("plots3/ncues_distplot.png")

    fig, ax = plt.subplots(figsize=((12, 12)))
    sns.scatterplot(mean_ncues, mean_corrects)
    ax.set(xlabel='number of requested cues before decision', ylabel='accuracy', xlim=((0, 6)))
    plt.savefig("plots3/accuracy_vs_ncues_scatter.png")


# def optimize_learning(max_evals=10, participant=0, n_trials_train=1, n_trials_test=1, seed=0):
#     hyperparams = {}
#     hyperparams['participant'] = participant
#     hyperparams['n_trials_train'] = n_trials_train
#     hyperparams['n_trials_test'] = n_trials_test
#     hyperparams['correct_early'] = 0  # hp.uniform('correct_early', 0, 0)
#     hyperparams['correct_late'] = hyperopt.hp.uniform('correct_late', 0, -1)
#     hyperparams['incorrect_early'] = hyperopt.hp.uniform('incorrect_early', 1, 0)
#     hyperparams['incorrect_late'] = 0  # hp.uniform('incorrect_late', 0, 0)
#     hyperparams['seed'] = seed

#     def objective(hyperparams):

#         participant = hyperparams['participant']
#         n_trials_train = hyperparams['n_trials_train']
#         n_trials_test = hyperparams['n_trials_test']
#         correct_early = hyperparams['correct_early']
#         correct_late = hyperparams['correct_late']
#         incorrect_early = hyperparams['incorrect_early']
#         incorrect_late = hyperparams['incorrect_late']
#         rng = np.random.RandomState(seed=hyperparams['seed'])

#         d_gain, loss = run_trials(
#             participant=participant,
#             n_trials_train=n_trials_train,
#             n_trials_test=n_trials_test,
#             correct_early=correct_early,
#             correct_late=correct_late,
#             incorrect_early=incorrect_early,
#             incorrect_late=incorrect_late,
#             rng=rng)

#         return {
#             'loss': loss,
#             'd_gain': d_gain,
#             'correct_early': correct_early,
#             'correct_late': correct_late,
#             'incorrect_early': incorrect_early,
#             'incorrect_late': incorrect_late,
#             'status': hyperopt.STATUS_OK }

#     trials = hyperopt.Trials()
#     hyperopt.fmin(objective,
#         rstate=np.random.RandomState(seed=seed),
#         space=hyperparams,
#         algo=hyperopt.tpe.suggest,
#         max_evals=max_evals,
#         trials=trials)
#     best_idx = np.argmin(trials.losses())
#     best = trials.trials[best_idx]
#     best_loss = best['result']['loss']
#     best_d_gain = best['result']['d_gain']
#     best_correct_late = best['result']['correct_late']
#     best_incorrect_early = best['result']['incorrect_early']
#     np.savez("best.npz", loss=best_loss, d_gain=best_d_gain, correct_late=best_correct_late, incorrect_early=best_incorrect_early)

#     return best_d_gain, best_correct_late, best_incorrect_early, best_loss


# plot_participant_data()

run_trials(participant=0, plot=True, strategy=0.25)
# run_trials(participant=11, plot=True, n_trials_train=5, n_trials_test=2, strategy=1.0)
