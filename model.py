import numpy as np
import nengo
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import nengolib
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
        if (len(np.where(data_range[:,2] > data_range[:,0])[0]) < len(data_range)/2 or
                len(np.where(data_range[:,2] > data_range[:,1])[0]) < len(data_range)/2):
            return ncues+1
    return 6

def is_correct(data, trial, dt=0.001):
    evidence_opt, choice_opt = get_evidence_opt(trial)
    for ncues in range(6):
        data_range = data[int(T*ncues/dt):int(T*(ncues+1)/dt)]
        # if decision has been A or B for more than half of the time in this slot, model has decided
        if len(np.where(data_range[:,0] > data_range[:,2])[0]) > len(data_range)/2:
            return choice_opt == 0
        elif len(np.where(data_range[:,1] > data_range[:,2])[0]) > len(data_range)/2:
            return choice_opt == 1
    best_end_utility = 0 if data[-1, 0] > data[-1, 1] else 1
    return best_end_utility == choice_opt

def get_ncues_opt(trial):
    # the time at which the evidence that could accumulate for option 2 can no longer outweight the current evidence for option 1
    # (regardless of actual cue values on this trial)
    evidence_opt, choice_opt = get_evidence_opt(trial)
    for n in range(6):
        possible_opposing_evidence = 0
        if evidence_opt[n][choice_opt] > evidence_opt[n][choice_opt-1] + np.sum(weights[n+1:]):
            return n+1

def get_ncues_opt_greedy(trial):
    # the time at which the evidence that will accumulate for option 2 can no longer outweight the current evidence for option 1
    # (given the actual cue values on this trial)
    evidence_opt, choice_opt = get_evidence_opt(trial)
    for n in range(6):
        if evidence_opt[n][choice_opt] > evidence_opt[-1][choice_opt-1]:
            return n+1


''' Model definition '''
weights = np.array([0.706, 0.688, 0.667, 0.647, 0.625, 0.6]) # weights of cues (from experiment)
T = 1.0  # time interval (s) for which the choices are presented

def run_model(trial, d_gain, t_train_start, t_train_end, k_train):

    values_A, values_B = read_values(trial)
    values = [[values_A[n], values_B[n]] for n in range(len(values_A))]

    # run PES learning with error depending on t_optimal vs. t_choose from previous trial
    def train_func(t, x):
        return k_train if t_train_start < t < t_train_end else 0

    constant_process = nengo.processes.PresentInput([[1]], presentation_time=6.0)
    weight_process = nengo.processes.PresentInput(weights, presentation_time=T)
    value_process = nengo.processes.PresentInput(values, presentation_time=T)

    model = nengo.Network(seed=0)
    with model:
        # Inputs
        weight_inpt = nengo.Node(weight_process)
        value_inpt = nengo.Node(value_process)
        time_inpt = nengo.Node(constant_process)
        train_inpt = nengo.Node(train_func, size_in=1)
        default_utility = nengo.Node(lambda t: 2 if t<6 else 0)
        weighted_value_node = nengo.Ensemble(1, 3, neuron_type=nengo.Direct())
        evidence_node = nengo.Ensemble(1, 2, neuron_type=nengo.Direct())
        
        # Ensembles
        time_cells = nengolib.networks.RollingWindow(theta=5.0, n_neurons=2000, dimensions=20,
            process=constant_process, neuron_type=nengo.LIF())  # LMU represents rolling window of input history
        gain = nengo.Ensemble(1000, 1, label="LC")  # represents emotional modulation of weights
        multiply = nengo.Ensemble(4000, 3, radius=4, label="dlPFC")  # represents values dim=[0,1] and weight dim=[2]
        evidence = nengo.Ensemble(4000, 2, radius=4)  # 2D integrator accumulates weighted evidence
        utility = nengo.Ensemble(4000, 3, radius=4, label="SMA")  # inputs to BG
        decision = nengo.networks.BasalGanglia(dimensions=3)  # WTA action selection between A, B, and more
        error = nengo.Ensemble(1000, 1, label="ACC")  # 'dopaminergic' error population for RL

        # Connections
        nengo.Connection(time_inpt, time_cells.input, synapse=None)
        nengo.Connection(value_inpt, multiply[0:2], synapse=None)
        nengo.Connection(weight_inpt, multiply[2], synapse=None)
        # define the delays (0s, 1s, 2s, 3s, 4s, 5s) for the LMU
        delays = time_cells.add_output(t=[0, 1.0/5, 2.0/5, 3.0/5, 4.0/5, 1], function=lambda w: w)  
        # learn the relationship between time and gain modulation using PES learning from previous trial's outcome
        conn_modulate = nengo.Connection(time_cells.state.neurons, gain, synapse=0.1,
            transform=d_gain.T, learning_rule_type=nengo.PES(learning_rate=1e-6))
        nengo.Connection(gain, multiply[2], synapse=0.1)
        # accumulate evidence for choice A and B by feeding weighted values into a 2D integrator
        # function multiplies represented weights by input values
        nengo.Connection(multiply, evidence[0], synapse=0.1, function=lambda x: x[0]*x[2], transform=0.1)
        nengo.Connection(multiply, evidence[1], synapse=0.1, function=lambda x: x[1]*x[2], transform=0.1)
        nengo.Connection(evidence, evidence, synapse=0.1)
        # compare the accumulated evidence for A and B against the default utility
        nengo.Connection(evidence, utility[0:2], synapse=0.1)
        nengo.Connection(default_utility, utility[2])
        # action selection via basal ganglia
        nengo.Connection(utility, decision.input)
        # externally computed error for PES learning (error = actual - target)
        nengo.Connection(train_inpt, error, synapse=None)
        nengo.Connection(error, conn_modulate.learning_rule, synapse=0.1)
        # ideal evidence accumulation (computed in math, compare to noisy integrator)
        nengo.Connection(value_inpt, weighted_value_node[0:2])
        nengo.Connection(weight_inpt, weighted_value_node[2])
        nengo.Connection(weighted_value_node, evidence_node,
            synapse=1/nengolib.signal.s, function=lambda x: [x[0]*x[2], x[1]*x[2]])

        # Probes
        p_weight_inpt = nengo.Probe(weight_inpt, synapse=None)
        p_value_inpt = nengo.Probe(value_inpt, synapse=None)
        p_time_cells = nengo.Probe(delays, synapse=0.1)
        p_gain = nengo.Probe(gain, synapse=0.1)
        p_multiply = nengo.Probe(multiply, synapse=0.1)
        p_evidence = nengo.Probe(evidence, synapse=0.1)
        p_utility = nengo.Probe(utility, synapse=0.1)
        p_decision = nengo.Probe(decision.output, synapse=0.1)  
        p_error = nengo.Probe(error, synapse=0.1)
        p_evidence_node = nengo.Probe(evidence_node, synapse=0.1)
        p_weights = nengo.Probe(conn_modulate, 'weights', synapse=None)

    sim = nengo.Simulator(model, seed=0)
    with sim:
        sim.run(6)

    return dict(
        times=sim.trange(),
        time_cells=sim.data[p_time_cells],
        gain=sim.data[p_gain],
        multiply=sim.data[p_multiply],
        evidence=sim.data[p_evidence],
        utility=sim.data[p_utility],
        decision=sim.data[p_decision],
        error=sim.data[p_error],
        evidence_node=sim.data[p_evidence_node],
        d_gain=sim.data[p_weights][-1].T,  # save weights on learned connection
        )


''' Experimental trials '''
n_trials = 20
d_gain = np.zeros((2000, 1))
t_train_start = 0
t_train_end = 0
k_train = 0
rng = np.random.RandomState(seed=1)
# table of rewards
correct_early = 0
correct_late = -1
incorrect_early = 1
incorrect_late = 0

for trial in range(n_trials):
    trial = rng.randint(40)
    print('trial: ', trial)
    data = run_model(trial, d_gain, t_train_start, t_train_end, k_train)

    # determine the timing and correctness of model behavior
    ncues_model = ncues(data['decision'])
    ncues_opt = get_ncues_opt(trial)
    error = ncues_opt - ncues_model
    correct = is_correct(data['decision'], trial)
    early = error >= 0
    if correct and early:
        k_train = correct_early
        t_train_start = ncues_model-1
        t_train_end = ncues_model
    if correct and not early:
        k_train = correct_late
        t_train_start = ncues_opt-1
        t_train_end = ncues_opt
    if not correct and early:
        k_train = incorrect_early
        t_train_start = ncues_model-1
        t_train_end = ncues_model
    if not correct and not early:
        k_train = incorrect_late
        t_train_start = ncues_opt-1
        t_train_end = ncues_opt
    if not correct and (ncues_model==6 or ncues_opt==6):
        k_train = 0
    # print("n cues empirical: ", [read_ncues_empirical(n, trial) for n in range(17)])

    # Plots
    fig, (ax1, ax2) = plt.subplots(2, 1, sharex=True, figsize=((12, 12)))
    # ax1.plot(data['times'], data['evidence'][:,0], label='evidence A', color='r')
    # ax1.plot(data['times'], data['evidence'][:,1], label='evidence B', color='b')
    # ax1.plot(data['times'], data['multiply'][:,0], label='mul A')
    # ax1.plot(data['times'], data['multiply'][:,1], label='mul B')
    ax1.plot(data['times'], data['utility'][:,0], label='utility A', color='r')
    ax1.plot(data['times'], data['utility'][:,1], label='utility B', color='b')
    ax1.plot(data['times'], data['evidence_node'][:,0], label='optimal A', color='r', linestyle="--")
    ax1.plot(data['times'], data['evidence_node'][:,1], label='optimal B', color='b', linestyle="--")
    ax1.plot(data['times'], data['utility'][:,2], label='utility more', color='k')
    # ax3.plot(data['times'], data['time_cells'], label='time_cells')
    # ax3.plot(data['times'], data['gain'], label='gain')
    ax2.plot(data['times'], data['error'], label='error')
    ax2.plot(data['times'], data['gain'], label='gain')
    ax2.axvline(x=ncues_model, alpha=0.5, label='ncues_model', color='r')
    ax2.axvline(x=ncues_opt, alpha=0.5, label='ncues_optimal', color='b')
    ax1.set(xticks=([0, 1, 2, 3, 4, 5, 6]), ylim=((0, 2.5)), title='trial %s, correct=%s'%(trial, correct))  # ylim=((0, np.sum(weights))),
    # ax3.set(xticks=([0, 1, 2, 3, 4, 5, 6]), ylim=((-0.2, 1.2)), xlabel='cues presented')
    ax2.set(xticks=([0, 1, 2, 3, 4, 5, 6]), ylim=((-0.1, 1.1)))
    ax1.legend(loc='upper left')
    ax2.legend(loc='upper left')
    # ax3.legend(loc='upper left')
    plt.savefig("plots/timeseries_%s.png"%trial)

    # update weights on learned connection for next trial
    d_gain = data['d_gain']