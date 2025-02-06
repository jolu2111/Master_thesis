# Plan

1. Find a problem that is related to probabilistic design in Geotechnics.

- Toy problem that are very simple

2. Solve the problem with a PINN where variables can have a range of values.

- E.g a consolidation problem, Terzaghi, with varying thickness and consolidation coefficient.

3. Use the PINN to compute a probability distribution of the point of interest

- This could be ultimate limit state (ULS) or some kind of serviceability limit state (SLS)

4. Finally use this distribution for a probabilistic design rather than Eurocode.

## Short term plan

1. Read about Bayesian neural networks.

   - Bayesian approach and probabilistic models.

2. Read on UQ_something - uncertainty quantification?

3. Read on Pyro - how to use it in probabilistic modelling

### Concrete Plans

1. Watch the "Probabilistic AI school" at youtube

2. Read the following papers in this order:

- https://arxiv.org/pdf/1505.05424
- https://www.sciencedirect.com/science/article/pii/S0021999120306872
- Pyro tutorials
- Look into Lower-level resources on Markov Chain Monte Carlo (MCMC), variational inference, and relevant Bayesian methods. These are crucial if you want to fully grasp the training loop behind B-PINNs.

3.

<hr style="border:2px solid red">

# Thoughts

- Have an overview of other machine learning models that can be used, when let's say you have data. This could be long short-term memory neural networks. The problem with this is that you need labelled data from field investigations, I don't have that.

- Use DeepXDE library when possible when dealing with difficult geometries.

- Soil property characterization is an area BNN could work.

# Questions

- I have read about BNN, but does not understand where they come into play if a PINN were to be used for a MC analysis. They are able to tell you the uncertainty of the model itself, but not the probability of something like SLS failure.

  - Is it that the BNN will say something about the models' uncertainty for ONE calculation, and the PINN will be able to calculate the problems' probability distribution?

<!-- - Should I focus on how a presumably well-trained PINN can be utilized for probabilistic design, and ignore the accuracy of the PINN itself, or should I also/ rather focus on the performance of the PINN or NN. Ref https://arxiv.org/pdf/2501.16371 that talks about what optimizers work the best.

    - Concentrate on the usage first, not a good model

- PINNs are hard to use for nonlinear problems such as for plastic behaviour of soil in a slope analysis. Should I stick to linear systems like a linear elastic? What type of problems should I look into?

    - It is not possible with PINNs to do a nonlinear analysis, but I can search up elasto plastic PINN if I want to. Probably to difficult for now.  -->
