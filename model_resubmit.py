import numpy as np
import nengo
import nengolib
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import nengolib
import scipy
sns.set(style="white", context="paper")

def go(values, validities, valence, thr=1.0, cert=0.2, urg=0.3, emo=0.5, seed=0):

    # Individual Parameters
    # thr : positive => slower decisions by default
    # cert :  positive => faster decisions vs delta
    # urg : larger => faster decisions vs time
    # emo : larger => faster decisions vs emotion
    # seed : nengo network seed

    # Fixed Parameters
    T = 1.0  # time interval (s) for which the choices are presented
    ramp = 1.0  # time cell ramping
    tau = 0.1  # synapse time constant
    bg = 0.4  # constant scaling of utility into BG

    # Inputs and Functions
    validity_process = nengo.processes.PresentInput(validities, presentation_time=T)
    value_process = nengo.processes.PresentInput(values, presentation_time=T)
    time_process = lambda t: t*urg  # time pressure ramps over each trial
    emotion_process = lambda t: valence*emo  # fixed input for pre-trial viewing of arousing images
    mult = lambda x: [x[0]*x[2], x[1]*x[2]]  # values weighted by remembered validity
    certainty = lambda x: cert*np.abs(x[0]-x[1]) - thr # certainty (scaled difference in accum. evidence)
    urgency = lambda x: [x[0]+x[1], x[0]+x[1]]  # total arousal (time+emotion) added to action utility
    thr_bias = lambda x: [-x[0], -x[0]]  # larger confidence (larger delta) lowers threshold

    # Model definition
    model = nengo.Network(seed=seed)
    model.config[nengo.Connection].synapse = nengo.Lowpass(0.05)
    model.config[nengo.Probe].synapse = nengo.Lowpass(0.05)

    with model:
        # Inputs
        reward_memory = nengo.Node(validity_process, label="reward memory")
        sensory_input = nengo.Node(value_process, label="sensory input")
        elapsed_time = nengo.Node(time_process, label="elapsed time")
        emotional_state = nengo.Node(emotion_process, label="emotional state")
     
        # Ensembles
        raw_value = nengo.Ensemble(1000, 2, radius=2, label="raw value (vision)")
        validity = nengo.Ensemble(1000, 1, label="validity ()")
        weighted_value = nengo.Ensemble(1000, 3, radius=3, label="weighted value (OFC)")
        evidence = nengo.Ensemble(2000, 2, radius=5, label="accum. evidence (PFC)")
        utility = nengo.Ensemble(2000, 2, radius=5, label="action utility (pSMA)")
        task_monitor = nengo.Ensemble(1000, 2, radius=2, label="task monitor (ACC)")
        arousal = nengo.Ensemble(1000, 2, radius=2, label="arousal (LC)")
        decision = nengo.networks.BasalGanglia(2, 200)
        motor_output = nengo.networks.Thalamus(2, 200, threshold=0.4)
        
        # Connections
        nengo.Connection(sensory_input, raw_value, synapse=None)
        nengo.Connection(reward_memory, validity, synapse=None)
        nengo.Connection(elapsed_time, task_monitor[1], synapse=None)
        nengo.Connection(emotional_state, arousal[0], synapse=None)
        nengo.Connection(validity, weighted_value[2])
        nengo.Connection(raw_value, weighted_value[:2])
        nengo.Connection(weighted_value, evidence, function=mult, synapse=tau, transform=tau)
        nengo.Connection(evidence, evidence, synapse=tau)
        nengo.Connection(evidence, utility)
        nengo.Connection(utility, decision.input, transform=bg)
        nengo.Connection(decision.output, motor_output.input)
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
        p_decision = nengo.Probe(decision.output)
        p_motor_output = nengo.Probe(motor_output.output)
        
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

    return(sim.data[p_motor_output])

def run_trails():
    # Read empirical data from .csv files
    trial = 2
    validities = np.array([0.706, 0.688, 0.667, 0.647, 0.625, 0.6]) # from experiment
    valence = 0.0  # emotional input
    def read_values(trial):
        stim = pd.read_csv('input_stimuli.csv', sep=';')
        trdata = list(stim[['c1','c2','c3','c4','c5','c6','c1.1','c2.1','c3.1','c4.1','c5.1','c6.1']].loc[trial])
        A = trdata[:len(trdata)//2]
        B = trdata[len(trdata)//2:]
        return[[A[n], B[n]] for n in range(len(A))]

    results = go(read_values(trial), validities, valence)
    fig, ax = plt.subplots()
    ax.plot(results)
    plt.savefig('test.pdf')

    # cues = pd.read_csv('how_many_cues.csv', sep=';')
    # choices = pd.read_csv('choices.csv', sep=';')

run_trails()
