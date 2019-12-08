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
        evidence_opt[n] = np.array([values_A[n], values_B[n]]) * weights[n] + evidence_opt[n-1]
    choice_opt = np.argmax(evidence_opt[-1])
    return evidence_opt, choice_opt

def ncues(data, dt=0.001):
    for ncues in range(6):
        data_range = data[int(T*ncues/dt):int(T*(ncues+1)/dt)]
        # if decision has been A or B for more than half of the time in this slot, model has decided
#         if (len(np.where(data_range[:,2] < data_range[:,0])[0]) > len(data_range)/2 or
#                 len(np.where(data_range[:,2] < data_range[:,1])[0]) > len(data_range)/2):
        # if decision is A or B at the end of this time slot, model has decided
        if data_range[:,0][-1] > data_range[:,2][-1] or data_range[:,1][-1] > data_range[:,2][-1]:
            return ncues+1
    return 6

def is_correct(data, trial, dt=0.001):
    evidence_opt, choice_opt = get_evidence_opt(trial)
    for ncues in range(6):
        data_range = data[int(T*ncues/dt):int(T*(ncues+1)/dt)]
        # find which dimension of 'decision' has been the most dominant over the window
        greatest = np.zeros((3))
        for t in range(len(data_range)):
            greatest[np.argmax(data_range[t])] += 1
        if np.argmax(greatest) == 0:
            return choice_opt == 0
        elif np.argmax(greatest) == 1:
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
        if evidence_opt[n][choice_opt] > evidence_opt[n][choice_opt-1] + np.sum(weights[n+1:]):
            return n+1

def get_ncues_oracle(trial):
    # the time at which the evidence that will accumulate for option 2 can no longer outweight the current evidence for option 1
    # (given the actual cue values on this trial)
    evidence_opt, choice_opt = get_evidence_opt(trial)
    for n in range(6):
        if evidence_opt[n][choice_opt] > evidence_opt[-1][choice_opt-1]:
            return n+1

# def get_ncues_greedy(trial):
#     # the first time at which the evidence accumulated for the winning option outweights the evidence accumulated for the non-winner
#     # (given the actual cue values on this trial)
#     evidence_opt, choice_opt = get_evidence_opt(trial)
#     for n in range(6):
#         if evidence_opt[n][choice_opt] > evidence_opt[n][choice_opt-1]:
#             return n+1
        
def get_ncues_greedy(trial):
    # the first time at which the evidence accumulated for one option outweights the evidence accumulated for the other option
    evidence_opt, choice_opt = get_evidence_opt(trial)
    for n in range(6):
        if evidence_opt[n][0] != evidence_opt[n][1]:
            return n+1


''' Model definition '''
weights = np.array([0.706, 0.688, 0.667, 0.647, 0.625, 0.6]) # weights of cues (from experiment)
T = 1.0  # time interval (s) for which the choices are presented

def run_model(trial, gain_in, default_in):

    values_A, values_B = read_values(trial)
    values = [[values_A[n], values_B[n]] for n in range(len(values_A))]

    weight_process = nengo.processes.PresentInput(weights, presentation_time=T)
    value_process = nengo.processes.PresentInput(values, presentation_time=T)

    model = nengo.Network(seed=0)
    with model:
        # Inputs
        weight_inpt = nengo.Node(weight_process)
        value_inpt = nengo.Node(value_process)
        gain_inpt = nengo.Node(gain_in)
        default_inpt = nengo.Node(default_in)
        weighted_value_node = nengo.Ensemble(1, 3, neuron_type=nengo.Direct())
        evidence_node = nengo.Ensemble(1, 2, neuron_type=nengo.Direct())
        
        # Ensembles
        gain = nengo.Ensemble(200, 1, radius=2, label="LC")  # represents emotional modulation of weights
        default = nengo.Ensemble(200, 1, radius=3)
        multiply = nengo.Ensemble(4000, 3, radius=4, label="dlPFC")  # represents values dim=[0,1] and weight dim=[2]
        evidence = nengo.Ensemble(4000, 2, radius=4)  # 2D integrator accumulates weighted evidence
        utility = nengo.Ensemble(4000, 3, radius=4, label="pre-SMA")  # inputs to BG
        decision = nengo.networks.BasalGanglia(n_neurons_per_ensemble=1000, dimensions=3)  # WTA action selection between A, B, and more

        # Connections
        nengo.Connection(gain_inpt, gain, synapse=None)
        nengo.Connection(default_inpt, default, synapse=None)
        nengo.Connection(value_inpt, multiply[0:2], synapse=None)
        nengo.Connection(weight_inpt, multiply[2], synapse=None)
        nengo.Connection(gain, multiply[2], synapse=0.1)
        # accumulate evidence for choice A and B by feeding weighted values into a 2D integrator
        # function multiplies represented weights by input values
        nengo.Connection(multiply, evidence[0], synapse=0.1, function=lambda x: x[0]*x[2], transform=0.1)
        nengo.Connection(multiply, evidence[1], synapse=0.1, function=lambda x: x[1]*x[2], transform=0.1)
        nengo.Connection(evidence, evidence, synapse=0.1)
        # compare the accumulated evidence for A and B against the default utility
        nengo.Connection(evidence, utility[0:2], synapse=0.1)
        nengo.Connection(default, utility[2])
        # action selection via basal ganglia
        nengo.Connection(utility, decision.input, synapse=0.1)
        # ideal evidence accumulation (computed in math, compare to noisy integrator)
        nengo.Connection(value_inpt, weighted_value_node[0:2])
        nengo.Connection(weight_inpt, weighted_value_node[2])
        nengo.Connection(weighted_value_node, evidence_node,
            synapse=1/nengolib.signal.s, function=lambda x: [x[0]*x[2], x[1]*x[2]])

        # Probes
        p_weight_inpt = nengo.Probe(weight_inpt, synapse=None)
        p_value_inpt = nengo.Probe(value_inpt, synapse=None)
        p_gain = nengo.Probe(gain, synapse=0.1)
        p_default = nengo.Probe(default, synapse=0.1)
        p_default = nengo.Probe(default, synapse=0.1)
        p_multiply = nengo.Probe(multiply, synapse=0.1)
        p_evidence = nengo.Probe(evidence, synapse=0.1)
        p_utility = nengo.Probe(utility, synapse=0.1)
        p_decision = nengo.Probe(decision.output, synapse=0.1)
        p_evidence_node = nengo.Probe(evidence_node, synapse=0.1)

    sim = nengo.Simulator(model, seed=0, progress_bar=True)
    with sim:
        sim.run(6, progress_bar=True)

    return dict(
        times=sim.trange(),
        gain=sim.data[p_gain],
        default=sim.data[p_default],
        multiply=sim.data[p_multiply],
        evidence=sim.data[p_evidence],
        utility=sim.data[p_utility],
        decision=sim.data[p_decision],
        evidence_node=sim.data[p_evidence_node],
        )

''' Experimental trials '''

def run_trials(
    participant=0,
    n_trials_train=48,
    n_trials_test=48,
    delta_gain=1e-1,
    delta_default=3e-1,
    strategy=1,  # 0 favors speed, 1 favors accuracy
    rng=np.random.RandomState(seed=0),
    plot=False):

    # Training
    gain = 0
    default = 0.5
    trial_list = np.arange(48)
    rng.shuffle(trial_list)
    for trial in range(n_trials_train):
        trial = trial_list[np.mod(trial, 48)]
        print('training trial: ', trial)
        data = run_model(trial, gain, default)

        # determine the timing and correctness of model behavior
        ncues_model = ncues(data['decision'])
        ncues_opt = get_ncues_opt(trial)
        ncues_greedy = get_ncues_greedy(trial)
        ncues_target = (strategy*ncues_opt + (1-strategy)*ncues_greedy)
        correct = is_correct(data['decision'], trial)
        print('correct:', correct==1)
        print('ncues_model:', ncues_model)
        print('ncues_opt', ncues_opt)
        print('ncues_greedy', ncues_greedy)
        print('ncues_target', ncues_target)

        if plot:
            fig, (ax1, ax2) = plt.subplots(2, 1, sharex=True, figsize=((12, 12)))
            ax1.plot(data['times'], data['utility'][:,0], label='utility A', color='r')
            ax1.plot(data['times'], data['utility'][:,1], label='utility B', color='b')
            ax1.plot(data['times'], data['evidence_node'][:,0], label='optimal A', color='r', linestyle="--")
            ax1.plot(data['times'], data['evidence_node'][:,1], label='optimal B', color='b', linestyle="--")
            ax1.plot(data['times'], data['default'], label='default utility', color='k')
            ax2.plot(data['times'], data['decision'][:,0], label='A')
            ax2.plot(data['times'], data['decision'][:,1], label='B')
            ax2.plot(data['times'], data['decision'][:,2], label='more')
            ax2.axvline(x=ncues_model, alpha=0.5, label='ncues_model', color='r')
            ax2.axvline(x=ncues_opt, alpha=0.5, label='ncues_opt', color='b')
            ax2.axvline(x=ncues_greedy, alpha=0.5, label='ncues_greedy', color='m')
            ax2.axvline(x=ncues_target, alpha=0.5, label='ncues_target', color='k')
            ax1.set(xticks=([0, 1, 2, 3, 4, 5, 6]), ylim=((0, 3.5)), ylabel='utility', title='trial %s, gain=%.3f, default=%.3f, correct=%s'%(trial, gain, default, correct))  # ylim=((0, np.sum(weights))),
            ax2.set(xticks=([0, 1, 2, 3, 4, 5, 6]), ylabel='BG values')
            ax1.legend(loc='upper left')
            ax2.legend(loc='upper left')
            plt.savefig("plots/timeseries_train_%s.png"%trial)
            plt.close()

        # update values for next trial
        if correct and ncues_model > ncues_target:  # correct and late
            gain += delta_gain * (1-strategy) * rng.uniform(0, 1)
            default -= delta_default * (1-strategy) * rng.uniform(0, 1)
        if not correct and ncues_model < ncues_target:  # incorrect and early
            gain -= delta_gain * strategy * rng.uniform(0, 1)
            default += delta_default * strategy * rng.uniform(0, 1)
        np.savez("data.npz", gain=gain, default=default, strategy=strategy, participant=participant, delta_gain=delta_gain, delta_default=delta_default)

    # Testing
    print("final: gain=%.3f, default=%.3f"%(gain, default))
    corrects_simulated = np.zeros((n_trials_test, 1))
    corrects_empirical = np.zeros((n_trials_test, 1))
    ncues_simulated = np.zeros((n_trials_test, 1))
    ncues_empirical = np.zeros((n_trials_test, 1))
    for trial in range(n_trials_test):
        print('testing trial: ', trial)
        data = run_model(trial, gain, default)

        ncues_model = ncues(data['decision'])
        ncues_opt = get_ncues_opt(trial)
        ncues_greedy = get_ncues_greedy(trial)
        ncues_target = (strategy*ncues_opt + (1-strategy)*ncues_greedy)
        correct = is_correct(data['decision'], trial)
        print('correct:', correct==1)
        print('ncues_model:', ncues_model)
        print('ncues_target', ncues_target)

        if plot:
            fig, (ax1, ax2) = plt.subplots(2, 1, sharex=True, figsize=((12, 12)))
            ax1.plot(data['times'], data['utility'][:,0], label='utility A', color='r')
            ax1.plot(data['times'], data['utility'][:,1], label='utility B', color='b')
            ax1.plot(data['times'], data['evidence_node'][:,0], label='optimal A', color='r', linestyle="--")
            ax1.plot(data['times'], data['evidence_node'][:,1], label='optimal B', color='b', linestyle="--")
            ax1.plot(data['times'], data['default'], label='default utility', color='k')
            ax2.plot(data['times'], data['decision'][:,0], label='A')
            ax2.plot(data['times'], data['decision'][:,1], label='B')
            ax2.plot(data['times'], data['decision'][:,2], label='more')
            ax2.axvline(x=ncues_model, alpha=0.5, label='ncues_model', color='r')
            ax2.axvline(x=ncues_opt, alpha=0.5, label='ncues_opt', color='b')
            ax2.axvline(x=ncues_greedy, alpha=0.5, label='ncues_greedy', color='m')
            ax2.axvline(x=ncues_target, alpha=0.5, label='ncues_target', color='k')
            ax1.set(xticks=([0, 1, 2, 3, 4, 5, 6]), ylim=((0, 3.5)), ylabel='utility', title='trial %s, correct=%s'%(trial, correct))  # ylim=((0, np.sum(weights))),
            ax2.set(xticks=([0, 1, 2, 3, 4, 5, 6]), ylabel='BG values')
            ax1.legend(loc='upper left')
            ax2.legend(loc='upper left')
            plt.savefig("plots/timeseries_test_%s.png"%trial)
            plt.close()

        corrects_simulated[trial] = is_correct(data['decision'], trial)
        ncues_simulated[trial] = ncues(data['decision'])
        corrects_empirical[trial] = is_correct_empirical(read_choice(participant, trial), trial)
        ncues_empirical[trial] = read_ncues_empirical(participant, trial)
        print("n_cues model: ", ncues_simulated[trial])
        print("n_cues empirical: ", ncues_empirical[trial])
        print("correct model: ", corrects_simulated[trial])
        print("correct empirical: ", corrects_empirical[trial])

    mean_model = 100*np.mean(corrects_simulated)
    mean_empirical = 100*np.mean(corrects_empirical)
    loss = scipy.stats.entropy(ncues_simulated, ncues_empirical)
    # if plot:
    fig, ax = plt.subplots(figsize=((12, 12)))
    sns.distplot(ncues_simulated, ax=ax, label='model, accuracy=%.1f%%'%mean_model, bins=[1,2,3,4,5,6], kde=False)
    sns.distplot(ncues_empirical, ax=ax, label='empirical, accuracy=%.1f%%'%mean_empirical, bins=[1,2,3,4,5,6], kde=False)
    ax.set(xlabel='number of requested cues before decision', ylabel='frequency', title="entropy=%.5f"%loss, xlim=((0, 6)))
    plt.legend()
    plt.savefig("plots/compare_participant%s.png"%participant)

    return gain, default


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
        plt.savefig("plots/participant%s.png"%participant)

    fig, ax = plt.subplots(figsize=((12, 12)))
    sns.barplot('correct', 'n_cues', data=df)
    plt.savefig("plots/accuracy_vs_ncues_all.png")

    fig, ax = plt.subplots(figsize=((12, 12)))
    sns.scatterplot(mean_ncues, mean_corrects)
    ax.set(xlabel='number of requested cues before decision', ylabel='accuracy', xlim=((0, 6)))
    plt.savefig("plots/accuracy_vs_ncues_participants.png")


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

run_trials(participant=0, plot=True, strategy=0)
# run_trials(participant=11, plot=True, strategy=1)
