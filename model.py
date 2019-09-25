import numpy as np
import nengo
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import nengolib

sns.set(style="white", context="talk")
# nengo.rc.set('progress', 'progress_bar', 'nengo.utils.progress.TerminalProgressBar')

weights = np.array([0.706, 0.688, 0.667, 0.647, 0.625, 0.6]) # weights of cues (from experiment)
T = 1.0  # time interval (s) for which the choices are presented

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

def weight_func(t):
    if t < T:
         return weights[0]
    elif t < 2*T:
         return weights[1]
    elif t < 3*T:
        return weights[2]
    elif t < 4*T:
        return weights[3]
    elif t < 5*T:
        return weights[4]
    elif t < 6*T:
        return weights[5]
    else:
        return 0

def ncues(data, dt=0.001):
    for ncues in range(6):
        data_range = data[int(T*ncues/dt):int(T*(ncues+1)/dt)]
        # if decision has been A or B for more than half of the time in this slot, model has decided
        if (len(np.where(data_range[:,2] > data_range[:,0])[0]) < len(data_range)/2 or
                len(np.where(data_range[:,2] > data_range[:,1])[0]) < len(data_range)/2):
            return ncues+1
    return 6

def is_correct(data, trial, dt=0.001):
    values_A, values_B = read_values(trial)
    evidence_opt = np.zeros((6, 2))
    for n in range(6):
        evidence_opt[n] = np.array([values_A[n], values_B[n]]) * weights[n] + evidence_opt[n-1]
    # print('evidence_opt', evidence_opt)
    choice_opt = np.argmax(evidence_opt[-1])
    # print('choice_opt', choice_opt)

    for ncues in range(6):
        data_range = data[int(T*ncues/dt):int(T*(ncues+1)/dt)]
        # if decision has been A or B for more than half of the time in this slot, model has decided
        if len(np.where(data_range[:,0] > data_range[:,2])[0]) > len(data_range)/2:
            return choice_opt == 0
        elif len(np.where(data_range[:,1] > data_range[:,2])[0]) > len(data_range)/2:
            return choice_opt == 1
    best_end_utility = 0 if data[-1, 0] > data[-1, 1] else 1
    return best_end_utility == choice_opt

def ncues_opt_1(trial):
    # if model had guessed after this many cues on this trial, it would have chosen correctly,
    # given the values of the subsequent attributes
    values_A, values_B = read_values(trial)
    evidence_opt = np.zeros((6, 2))
    for n in range(6):
        evidence_opt[n] = np.array([values_A[n], values_B[n]]) * weights[n] + evidence_opt[n-1]
    # print('evidence_opt', evidence_opt)
    choice_opt = np.argmax(evidence_opt[-1])
    # print('choice_opt', choice_opt)
    ncues_opt = 6
    for n in np.arange(1, 7):
        # print(evidence_opt[-n])
        if np.argmax(evidence_opt[-n]) == choice_opt and np.sum(evidence_opt[-n]) > 0:
            ncues_opt = 7-n
    # print('ncues_opt', ncues_opt)
    return ncues_opt

def ncues_opt_2(trial):
    # if model had guess after this many cues on this trial, it could not have chosen correctly,
    # regardless of the values of the subsequent attributes
    values_A, values_B = read_values(trial)
    evidence_opt = np.zeros((6, 2))
    for n in range(6):
        evidence_opt[n] = np.array([values_A[n], values_B[n]]) * weights[n] + evidence_opt[n-1]
    choice_opt = np.argmax(evidence_opt[-1])
    ncues_opt = 6
    for n in np.arange(1, 7):
        # print(evidence_opt[-n][choice_opt], np.sum(weights[-n:]))
        if evidence_opt[-n][choice_opt] > np.sum(weights[-n:]):
            ncues_opt = 7-n
    return ncues_opt

def run_model(trial, d_gain, t_train, k_train):

    values_A, values_B = read_values(trial)
    print("values_A: ", values_A)
    print("values_B: ", values_B)

    def value_func(t):
        if t < T:
             return values_A[0], values_B[0]
        elif t < 2*T:
             return values_A[1], values_B[1]
        elif t < 3*T:
            return values_A[2], values_B[2]
        elif t < 4*T:
            return values_A[3], values_B[3]
        elif t < 5*T:
            return values_A[4], values_B[4]
        elif t < 6*T:
            return values_A[5], values_B[5]
        else:
            return 0, 0

    # run PES learning with error depending on t_optimal vs. t_choose from previous trial
    def train_func(t, x):
        if t_train-1 < t < t_train:
            return k_train
        else:
            return 0

    # inputs = [[1.0/6], [2.0/6], [3.0/6], [4.0/6], [5.0/6], [6.0/6]]
    process = nengo.processes.PresentInput([[1]], presentation_time=6.0)

    model = nengo.Network(seed=0)
    with model:
        # inputs
        weight_inpt = nengo.Node(weight_func)
        value_inpt = nengo.Node(value_func)
        time_inpt = nengo.Node(process)
        train_inpt = nengo.Node(train_func, size_in=1)
        default_utility = nengo.Node(lambda t: 2 if t<6 else 0)
        
        # ensembles
        weight = nengo.Ensemble(1000, 1, radius=2, label="OFC")  # represents value of current weight
        time_cells = nengolib.networks.RollingWindow(theta=5.0, n_neurons=2000, process=process, dimensions=20, neuron_type=nengo.LIF())
        pressure = nengo.Ensemble(2000, 6, radius=3, label="ACC")  # represents emotional state
        gain = nengo.Ensemble(1000, 1, label="LC")  # represents emotional modulation of weights
        multiply = nengo.Ensemble(2000, 3, radius=4, label="dlPFC")  # multiplies represented weights by input values
        evidence = nengo.Ensemble(2000, 2, radius=4)  # 2D integrator accumulates weighted evidence
        utility = nengo.Ensemble(2000, 3, radius=4, label="SMA")  # inputs to BG
        decision = nengo.networks.BasalGanglia(dimensions=3)  # WTA action selection between A, B, and more
        error = nengo.Ensemble(1000, 1)  # error = t_optimal - t_choose (nonzero after choice is made)

        # connections
        nengo.Connection(weight_inpt, weight, synapse=None)
        nengo.Connection(time_inpt, time_cells.input, synapse=None)
        nengo.Connection(value_inpt, multiply[0:2], synapse=None)
        nengo.Connection(weight, multiply[2])
        # learn the relationship between pressure and gain modulation using PES learning from previous trial's outcome
        # conn_modulate = nengolib.Connection(time_cells.add_output(t=0, function=lambda w: w), pressure, synapse=0.1)
        nengolib.Connection(time_cells.add_output(t=[0, 1.0/5, 2.0/5, 3.0/5, 4.0/5, 1], function=lambda w: w),
            pressure, synapse=0.1)
        conn_modulate = nengo.Connection(pressure.neurons, gain, synapse=0.1,
            # function=lambda x: 0,
            transform=d_gain.T,
            # solver=nengo.solvers.NoSolver(d_gain))
            learning_rule_type = nengo.PES(learning_rate=1e-6))
        nengo.Connection(gain, weight, synapse=0.1)
        # accumulate evidence for choice A and B by feeding weighted values into a 2D integrator
        nengo.Connection(multiply, evidence[0], synapse=0.1, function=lambda x: x[0]*x[2], transform=0.1)
        nengo.Connection(multiply, evidence[1], synapse=0.1, function=lambda x: x[1]*x[2], transform=0.1)
        nengo.Connection(evidence, evidence, synapse=0.1)
        # current difference in accumulated evidence for A and B contributes to pressure
        # nengo.Connection(evidence, pressure[1], function=lambda x: np.abs(x[0] - x[1]))  #todo: easier to represent func/
        # compare the accumulated evidence for A and B against the default utility
        nengo.Connection(evidence, utility[0:2], synapse=0.1)
        nengo.Connection(default_utility, utility[2])
        # action selection via basal ganglia and thalamus
        nengo.Connection(utility, decision.input)
        # input externally computed error for PES learning (error = actual - target)
        nengo.Connection(train_inpt, error, synapse=None)
        nengo.Connection(error, conn_modulate.learning_rule, synapse=0.1)

        # probes
        p_weight_inpt = nengo.Probe(weight_inpt, synapse=None)
        p_value_inpt = nengo.Probe(value_inpt, synapse=None)
        p_weight = nengo.Probe(weight, synapse=0.1)
        # p_time_cells = nengo.Probe(time_cells.add_output(t=[1], function=lambda w: [w[0]]), synapse=0.1)
        p_pressure = nengo.Probe(pressure, synapse=0.1)  # solver=nengo.solvers.NoSolver(d_gain), 
        p_gain = nengo.Probe(gain, synapse=0.1)
        p_multiply = nengo.Probe(multiply, synapse=0.1)
        p_evidence = nengo.Probe(evidence, synapse=0.1)
        p_utility = nengo.Probe(utility, synapse=0.1)
        p_decision = nengo.Probe(decision.output, synapse=0.1)  
        p_error = nengo.Probe(error, synapse=0.1)
        p_weights = nengo.Probe(conn_modulate, 'weights', synapse=None)

    sim = nengo.Simulator(model, seed=0)
    with sim:
        sim.run(6)

    return dict(
        times=sim.trange(),
        weight_inpt=sim.data[p_weight_inpt],
        value_inpt=sim.data[p_value_inpt],
        weight=sim.data[p_weight],
        # time_cells=sim.data[p_time_cells],
        pressure=sim.data[p_pressure],
        gain=sim.data[p_gain],
        multiply=sim.data[p_multiply],
        evidence=sim.data[p_evidence],
        utility=sim.data[p_utility],
        decision=sim.data[p_decision],
        error=sim.data[p_error],
        d_gain=sim.data[p_weights][-1].T,
        )

n_trials = 40
d_gain = np.zeros((2000, 1))
t_train = 0
k_train = 0
rng = np.random.RandomState(seed=1)
correct_early = 0
correct_late = -1
incorrect_early = -1
incorrect_late = -0.5

for t in range(n_trials):
    # trial = rng.randint(0, 40)
    trial = t
    print('trial: ', trial)
    k_train = 0
    data = run_model(trial, d_gain, t_train, k_train)

    ncues_model = ncues(data['decision'])
    ncues_opt1 = ncues_opt_1(trial)
    ncues_opt2 = ncues_opt_2(trial)
    error_1 = ncues_opt1 - ncues_model
    error_2 = ncues_opt2 - ncues_model
    correct = is_correct(data['decision'], trial)
    early = error_2 >= 0
    if correct and early:
        k_train = correct_early
    if correct and not early:
        k_train = correct_late
    if not correct and early:
        k_train = incorrect_early
    if not correct and not early:
        k_train = incorrect_late
    # if not correct and ncues_model == 6:
    #     k_train = 0
    t_train = np.min([ncues_model, ncues_opt2])
    print('k_train', k_train)
    print('t_train', t_train)

    print("n cues empirical: ", [read_ncues_empirical(n, trial) for n in range(17)])
    print("n cues model: ", ncues_model)
    print("n cues optimal_1: ", ncues_opt1)
    print("n cues optimal_2: ", ncues_opt2)

    fig, (ax1, ax3, ax2) = plt.subplots(3, 1, sharex=True, figsize=((12, 12)))
    # ax1.plot(data['times'], data['evidence'][:,0], label='evidence A')
    # ax1.plot(data['times'], data['evidence'][:,1], label='evidence B')
    # ax1.plot(data['times'], data['multiply'][:,0], label='mul A')
    # ax1.plot(data['times'], data['multiply'][:,1], label='mul B')
    ax2.plot(data['times'], data['multiply'][:,2], label='weight')
    ax1.plot(data['times'], data['utility'][:,0], label='utility A')
    ax1.plot(data['times'], data['utility'][:,1], label='utility B')
    ax1.plot(data['times'], data['utility'][:,2], label='utility more')
    # ax3.plot(data['times'], data['decision'][:,0], label='decision A')
    # ax3.plot(data['times'], data['decision'][:,1], label='decision B')
    # ax3.plot(data['times'], data['decision'][:,2], label='decision more')
    ax3.plot(data['times'], data['pressure'], label='pressure')
    ax3.plot(data['times'], data['gain'], label='gain')
    ax2.plot(data['times'], data['error'], label='error')
    # ax2.plot(data['times'], data['pressure'], label='pressure')
    ax2.plot(data['times'], data['gain'], label='gain')
    ax2.axvline(x=ncues_model, alpha=0.5, label='ncues_model', color='r')
    ax2.axvline(x=ncues_opt2, alpha=0.5, label='ncues_opt1', color='b')
    ax1.set(xticks=([0, 1, 2, 3, 4, 5, 6]), ylim=((0, 3.0)), title='trial %s, correct=%s'%(trial, correct))
    ax3.set(xticks=([0, 1, 2, 3, 4, 5, 6]), ylim=((-0.2, 1.2)), xlabel='cues presented')
    ax2.set(xticks=([0, 1, 2, 3, 4, 5, 6]), ylim=((-1, 1.5)))
    ax1.legend(loc='upper left')
    ax2.legend(loc='upper left')
    ax3.legend(loc='upper left')
    plt.savefig("plots/timeseries_%s.png"%t)

    d_gain = data['d_gain']