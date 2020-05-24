import numpy as np
import nengo
import nengolib
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import nengolib
import scipy
from hyperopt import fmin, tpe, hp, STATUS_OK, Trials, base
sns.set(style="white", context="paper")

def go(values, validities, valence,
        thr=1.0, cert=0.2, urg=0.3, emo=0.5,
        ramp=1.0, tau=0.1, bg=0.2, seed=0):

    # Parameters
    # thr : positive => slower decisions by default
    # cert :  positive => faster decisions vs delta
    # urg : larger => faster decisions vs time
    # emo : larger => faster decisions vs emotion
    # seed : nengo network seed
    # ramp : time cell ramping (fixed)
    # tau : synapse time constant (fixed)
    # bg : constant scaling of utility into BG (fixed)

    # Inputs and Functions
    validity_process = nengo.processes.PresentInput(validities, presentation_time=1.0)
    value_process = nengo.processes.PresentInput(values, presentation_time=1.0)
    time_process = lambda t: t*urg  # time pressure ramps over each trial, escalates before end
    force_process = lambda t: thr*(t>5.75)  # force high certainty near trial end to produce decision
    emotion_process = lambda t: valence*emo  # fixed input for pre-trial viewing of arousing images
    mult = lambda x: [x[0]*x[2], x[1]*x[2]]  # values weighted by remembered validity
    certainty = lambda x: cert*np.abs(x[0]-x[1]) - thr  # certainty (scaled difference in accum. evidence)
    urgency = lambda x: [x[0]+x[1], x[0]+x[1]]  # total arousal (time+emotion) added to action utility
    thr_bias = lambda x: [-x[0], -x[0]]  # larger confidence (larger delta) lowers threshold

    # Model definition
    model = nengo.Network(seed=seed)
    model.config[nengo.Connection].synapse = nengo.Lowpass(0.05)
    model.config[nengo.Probe].synapse = nengo.Lowpass(0.05)
    model.config[nengo.Probe].sample_every = 0.01

    with model:
        # Inputs
        reward_memory = nengo.Node(validity_process, label="reward memory")
        sensory_input = nengo.Node(value_process, label="sensory input")
        elapsed_time = nengo.Node(time_process, label="elapsed time")
        force_choice = nengo.Node(force_process, label="force choice")
        emotional_state = nengo.Node(emotion_process, label="emotional state")
     
        # Ensembles
        raw_value = nengo.Ensemble(2000, 2, radius=2, label="raw value (vision)")
        validity = nengo.Ensemble(2000, 1, label="validity ()")
        weighted_value = nengo.Ensemble(2000, 3, radius=4, label="weighted value (OFC)")
        evidence = nengo.Ensemble(2000, 2, radius=6, label="accum. evidence (PFC)")
        utility = nengo.Ensemble(2000, 2, radius=8, label="action utility (pSMA)")
        task_monitor = nengo.Ensemble(2000, 2, radius=2, label="task monitor (ACC)")
        arousal = nengo.Ensemble(1000, 2, radius=2, label="arousal (LC)")
        decision = nengo.networks.BasalGanglia(2, 200)
        motor = nengo.networks.Thalamus(2, 200, threshold=0.3)
        
        # Connections
        nengo.Connection(sensory_input, raw_value, synapse=None)
        nengo.Connection(reward_memory, validity, synapse=None)
        nengo.Connection(force_choice, task_monitor[0], synapse=None)
        nengo.Connection(elapsed_time, task_monitor[1], synapse=None)
        nengo.Connection(emotional_state, arousal[0], synapse=None)
        nengo.Connection(validity, weighted_value[2])
        nengo.Connection(raw_value, weighted_value[:2])
        nengo.Connection(weighted_value, evidence, function=mult, synapse=tau, transform=tau)
        nengo.Connection(evidence, evidence, synapse=tau)
        nengo.Connection(evidence, utility)
        nengo.Connection(utility, decision.input, transform=bg)
        nengo.Connection(decision.output, motor.input)
        nengo.Connection(evidence, task_monitor[0], function=certainty)
        nengo.Connection(task_monitor[1], arousal[1])
        nengo.Connection(arousal, utility, function=urgency)
        nengo.Connection(task_monitor, decision.stn.input, function=thr_bias)
        
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
        p_stn = nengo.Probe(decision.stn.output)
        p_decision = nengo.Probe(decision.output)
        p_motor = nengo.Probe(motor.output)
        
        # Ideal
        ideal_evidence = nengo.Network(label="ideal")
        with ideal_evidence:
            val = nengo.Ensemble(1, 3, neuron_type=nengo.Direct())
            ev = nengo.Ensemble(1, 2, neuron_type=nengo.Direct(), label="ideal")
            nengo.Connection(sensory_input, val[:2], synapse=None)
            nengo.Connection(reward_memory, val[2], synapse=None)
            nengo.Connection(val, ev, synapse=1/nengolib.signal.s, function=mult)
            p_ideal = nengo.Probe(ev, synapse=None)

    with nengo.Simulator(model, seed=seed, progress_bar=False) as sim:
        sim.run(6, progress_bar=True)

    return dict(
        times = sim.trange()[::10],
        raw_value = sim.data[p_raw_value],
        weighted_value = sim.data[p_weighted_value],
        evidence = sim.data[p_evidence],
        utility = sim.data[p_utility],
        task_monitor = sim.data[p_task_monitor],
        arousal = sim.data[p_arousal],
        stn = sim.data[p_stn],
        decision = sim.data[p_decision],
        motor = sim.data[p_motor],
        ideal = sim.data[p_ideal],
        )

def run_trails(thr, cert, urg, emo, n_trials=10, plot=True):
    # Read empirical data from .csv files
    validities = np.array([0.706, 0.688, 0.667, 0.647, 0.625, 0.6]) # from experiment
    valence = 1.0  # emotional input
    def read_values(trial):
        stim = pd.read_csv('input_stimuli.csv', sep=';')
        trdata = list(stim[['c1','c2','c3','c4','c5','c6','c1.1','c2.1','c3.1','c4.1','c5.1','c6.1']].loc[trial])
        A = trdata[:len(trdata)//2]
        B = trdata[len(trdata)//2:]
        return[[A[n], B[n]] for n in range(len(A))]

    RTs = np.zeros((n_trials))
    corrects = np.zeros((n_trials))
    for trial in range(n_trials):
        print('trial %s'%trial)
        data = go(thr, cert, urg, emo, read_values(trial), validities, valence)
        RT = get_RT(data['motor'])
        choice = get_choice(data['motor'])
        ch = "A" if choice==0 else "B"
        cr = "A" if data['ideal'][-1,0] > data['ideal'][-1,1] else "B"
        chc = ch + " (correct)" if ch==cr else ch + " (incorrect)"
        RTs[trial] = RT
        corrects[trial] = ch==cr
        if plot:
            fig, ax = plt.subplots()
            ax.plot(data['times'], data['ideal'][:,0], label=r"$A*$", color='r', linestyle="--")
            ax.plot(data['times'], data['ideal'][:,1], label=r"$B*$", color='b', linestyle="--")
            ax.plot(data['times'], data['evidence'][:,0], label=r"$\~{A}$", color='r', linestyle=":")
            ax.plot(data['times'], data['evidence'][:,1], label=r"$\~{B}$", color='b', linestyle=":")
            ax.plot(data['times'], data['utility'][:,0], label=r"$\hat{A}$", color='r', linestyle="-")
            ax.plot(data['times'], data['utility'][:,1], label=r"$\hat{B}$", color='b', linestyle="-")
            ax.plot(data['times'], data['motor'][:,0], label=r"$A_{out}$", color='r', linestyle="-.")
            ax.plot(data['times'], data['motor'][:,1], label=r"$B_{out}$", color='b', linestyle="-.")
            ax.set(xlabel="time(s)", ylabel="state", title="choice = %s \n RT = %.1fs"%(chc, RT))
            ax.legend()
            plt.savefig('trial_%s.pdf'%trial)

            fig, ax = plt.subplots()
            ax.plot(data['times'], data['task_monitor'][:,0], label=r"$conf$")
            ax.plot(data['times'], data['task_monitor'][:,1], label=r"$t$")
            ax.plot(data['times'], data['stn'][:,0], label=r"$A_{stn}$")
            ax.plot(data['times'], data['stn'][:,1], label=r"$B_{stn}$")
            ax.set(xlabel="time(s)", ylabel="state")
            ax.legend()
            plt.savefig('extra_%s.pdf'%trial)

            plt.close('all')

    return RTs, corrects

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

def compare_RTs(model, empirical, n_trials=10):
    entropy = scipy.stats.entropy(model, empirical)
    fig, ax = plt.subplots()
    ax.hist(model, density=True, bins=np.array([1,2,3,4,5,6,7])-0.5, rwidth=1, alpha=0.5, label='model')
    ax.hist(empirical, density=True, bins=np.array([1,2,3,4,5,6,7])-0.5, rwidth=1, alpha=0.5, label="empirical")
    ax.set(xlabel="RT", ylabel="Frequency", title="Entropy = %.5f"%entropy)
    plt.savefig("RTs_compare.pdf")
    return

def optimize_to_dist(target, n_trials=10, evals=3, seed=0):
    hyp = {}
    hyp['thr'] = hp.uniform('thr', -1, 0)
    hyp['cert'] = hp.uniform('cert', 0, 1)
    hyp['urg'] = hp.uniform('urg', 0, 1)
    # hyp['emo'] = hp.uniform('emo', 0, 1)
    hyp['emo'] = 0
    rng = np.random.RandomState(seed=seed)

    def objective(hyp):
        thr, cert, urg, emo = hyp['thr'], hyp['cert'], hyp['urg'], hyp['emo']
        RTs, corrects = run_trails(thr, cert, urg, emo, n_trials=n_trials, plot=False)
        loss = scipy.stats.entropy(RTs, target)
        return {'loss': loss, 'thr': thr, 'cert': cert, 'urg': urg, 'emo': emo, 'status': STATUS_OK}
    
    trials = Trials()
    fmin(objective, rstate=rng, space=hyp, algo=tpe.suggest, max_evals=evals, trials=trials)
    best_idx = np.argmin(trials.losses())
    best = trials.trials[best_idx]
    thr = best['result']['thr']
    cert = best['result']['cert']
    urg = best['result']['urg']
    emo = best['result']['emo']
    final_RTs, final_corrects = run_trails(thr, cert, urg, emo, n_trials=n_trials, plot=True)
    np.savez("best.npz", final_RTs=final_RTs, final_corrects=final_corrects,
        thr=thr, cert=cert, urg=urg, emo=emo)

n_trials = 10
participant = 0
# RTs_model, corrects_model = run_trails(n_trials=n_trials)
cues = pd.read_csv('how_many_cues.csv', sep=';')
RTs_emp = np.array(cues[cues.columns[participant+1]])[:n_trials]
optimize_to_dist(RTs_emp, n_trials=n_trials, evals=3)
# compare_RTs(RTs_model, RTs_emp, n_trials)