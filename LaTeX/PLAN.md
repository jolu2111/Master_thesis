# First Plan

1. Choose a simple, yet interesting PDE that has a parameter with a probability distribution, so the solution also has a probability distribution. 

2. Solve this PDE using a PINN with Monte Carlo dropout at training and inference (and/or other methods to get a probability distribution) in two ways:
- By fixing the input feature at a mean, and thereby only calculating the model uncertainty with MC dropout.
- Sampling the input feature from the probability distribution at every training-loop AND inference loop, and thereby calculating both model uncertainty and problem uncertainty with MC dropout. 
\
This will give the function pi(x).

3. Find the quasi-optimal PDF, h(X), using ...

4. Find the correction factor α<sub>corr</sub> from a known analytical solution or from FEM by comparing pi(x) with the real PDF f(x).

5. Final probability of failure is then α<sub>corr</sub> * Pf_e

<!-- # 
1. Find a problem that is related to probabilistic design in Geotechnics.

- Toy problem that are very simple

2. Solve the problem with a PINN where variables can have a range of values.

- E.g a consolidation problem, Terzaghi, with varying thickness and consolidation coefficient.

3. Use the PINN to compute a probability distribution of the point of interest

- This could be ultimate limit state (ULS) or some kind of serviceability limit state (SLS)

4. Finally use this distribution for a probabilistic design rather than Eurocode. -->

## Short term plan

1. DONE Implement normalization of the input to see if that helps. Look at chat GPT answer of 10.03.

2. Fix the parameterized PINN so it is only one class - as in the GitHub code.

3. FORM - implementation of the code.
  - SORM if time.

4. Find and use the CoV - coefficient of variation.

5. Have a look at the metamodel-based sampling again and see if I can implement some of it.

6. Review what Ivan has uploaded to Teams.

7. Read adaptive IS from Ivan.

8. Go through the code and minimize the amount of hyperparameters that must be chosen.

9. Find some improvement in the way the new points are chosen (e.g., clustering).

10. Find some improvement in the way the new points are used (e.g., partial substitution).

11. Implement PINN-FORM as in the paper, where the FORM is done inside the PINN as inverse calculation. 

11. Read on UQ (uncertainty quantification) to deepen understanding.

<!-- ### Concrete Plans

2. Read the following papers in this order:

- https://arxiv.org/pdf/1505.05424
- https://www.sciencedirect.com/science/article/pii/S0021999120306872
- Pyro tutorials
- Look into Lower-level resources on Markov Chain Monte Carlo (MCMC), variational inference, and relevant Bayesian methods. These are crucial if you want to fully grasp the training loop behind B-PINNs. -->

3.

<hr style="border:2px solid red">

# Thoughts


- Make a PINN that samples new points every so ans so iteration and compare the result with one that does not 
- Make three different approaches of finding the MPP 
  - FORM
  - LBFGS or Adam
  - Simple MC scan to find points
  
- OR sample points directly without going through the MPP: 
  - Subset simulation 
  - 

- Problem is that if you focus training on one part of the paramters, the ones left out gets very bad very quickly. Should in other words not disregard other areas completely.

- I can speed up the adaptive sampling massively if I only check for certain time points as in the training, rather then the whole time domain. 
-   But then I cant find if it dips below -1.0, since I need the whole time series to find that. 


# Questions

- For parameterized PINN: are y0 and v0 spatiotemporal coordinate (x, t) or PDE parameters µ with hparam = gθp(µ)?

- How should I organize my thesis - first talk about uncertainty quantification and then talk about PINNs, or should I do it all in one go. 

- FORM requires PINN to be accurate for a lot more of the paramterspace then crude monte carlo, since it works by using the derivatives, and that is my initial thought. Or am I wrong? 
  - ANSWER, yes, but you can have a second algorithm that walks along the g(0) line and tryes to get the distance shorter. 

- Should time also be a part of the LHS? I am currently first having t and params randomnlt distributed, and the adaptive scheme is improving the choise of params, but the time is still randomly chosen. 

<!-- - Should I focus on how a presumably well-trained PINN can be utilized for probabilistic design, and ignore the accuracy of the PINN itself, or should I also/ rather focus on the performance of the PINN or NN. Ref https://arxiv.org/pdf/2501.16371 that talks about what optimizers work the best.

    - Concentrate on the usage first, not a good model
-->

<!-- - PINNs are hard to use for nonlinear problems such as for plastic behaviour of soil in a slope analysis. Should I stick to linear systems like a linear elastic? What type of problems should I look into?

    - It is not possible with PINNs to do a nonlinear analysis, but I can search up elasto plastic PINN if I want to. Probably too difficult for now.  -->
