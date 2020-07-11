# Abstract

Decision making (DM) requires the coordination of anatomically and functionally distinct cortical and subcortical areas. While previous computational models have studied these subsystems in isolation, few models explore how DM holistically arises from their interaction. We propose a spiking neuron model that unifies various components of DM, then show that the model performs an inferential decision task in a human-like manner. The model (a) includes populations corresponding to dorsolateral prefrontal cortex, orbitofrontal cortex, right inferior frontal cortex, pre-supplementary motor area, and basal ganglia; (b) is constructed using 8000 leaky-integrate-and-fire neurons with 7 million connections; and (c) realizes dedicated cognitive operations such as weighted valuation of inputs, accumulation of evidence for multiple choice alternatives, competition between potential actions, dynamic thresholding of behavior, and urgency-mediated modulation. We show that the model reproduces reaction time distributions and speed-accuracy tradeoffs from humans performing the task. These results provide behavioral validation for tasks that involve slow dynamics and perceptual uncertainty; we conclude by discussing how additional tasks, constraints, and metrics may be incorporated into this initial framework.

# Installation and Simulation

 - pip install numpy scipy nengo matplotlib seaborn pandas nengolib jupyter

 - python model_resubmit.py

 - nengo gui_resubmit.py

  -  jupyter notebook plotter.ipynb
