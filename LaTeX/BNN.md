Below is a deliberately **small, hands-on example** of a Bayesian Neural Network (BNN) “by hand.” We’ll keep the network *extremely* simple—a **single-layer linear model** (i.e. one weight \(w\) and one bias \(b\) only). Then we’ll run **one iteration** of **Variational Inference** with a toy dataset of **two points**. This will demonstrate the **essence** of how a BNN updates its posterior distribution given data, without drowning in large matrices.

> **Disclaimer**: Real BNNs have *many* weights and require many iterations of gradient-based optimization or MCMC. This toy example is just to illustrate the mechanics step by step with real numbers.

---

# 1. Model Setup

### 1.1 The Network
We choose a single-input single-output neural net **with no hidden layers**:
\[
\text{NN}(x; w, b) = w \cdot x + b.
\]

This is effectively **linear regression**, but you can think of it as the simplest possible “neural net.”

### 1.2 The Data
We have **two observed points**:

\[
\begin{cases}
x_1 = 1, & y_1 = 1.8 \\
x_2 = 3, & y_2 = 2.7
\end{cases}
\]

### 1.3 Prior on the Weights
We place **independent Gaussian priors** on \(w\) and \(b\):

\[
w \sim \mathcal{N}(0,\;1^2) 
\quad\text{and}\quad
b \sim \mathcal{N}(0,\;1^2).
\]

### 1.4 Likelihood
We assume the observation noise is Gaussian with standard deviation \(\sigma = 0.2\). Formally,

\[
y_i \mid (w,b) \;\sim\; \mathcal{N}\!\bigl(w\,x_i + b,\; \sigma^2\bigr),
\quad \text{with } \sigma = 0.2.
\]

Hence the **data likelihood** for a single \((x_i, y_i)\) is

\[
p(y_i \mid w,b)
\;=\;
\frac{1}{\sqrt{2\pi}\,\sigma} \exp\!\Bigl(-\tfrac{1}{2\sigma^2}\bigl[y_i - (w\,x_i + b)\bigr]^{2}\Bigr).
\]

---

# 2. Variational Approximation

We want the **posterior** \(p(w,b \mid D)\), where \(D\) is our dataset. Doing so exactly can be messy, so we do **variational inference** (VI).

### 2.1 The Variational Family
We define a **Gaussian variational distribution** (a “surrogate” to approximate the posterior):
\[
q(w,b) = q(w)\,q(b),
\]
with
\[
w \sim \mathcal{N}(\mu_{w},\;s_{w}^2),
\quad
b \sim \mathcal{N}(\mu_{b},\;s_{b}^2).
\]
where \(\mu_{w}, \mu_{b}, s_{w}, s_{b}\) are *variational parameters* we will optimize.  
For simplicity, let’s start them at:
\[
\mu_{w}^{(0)}=0,\quad \mu_{b}^{(0)}=0,
\quad s_{w}^{(0)}=1,\quad s_{b}^{(0)}=1.
\]
(This is just an initial guess.)

### 2.2 Evidence Lower Bound (ELBO)

The **ELBO** is:
\[
\text{ELBO}(\mu_w, s_w, \mu_b, s_b) 
= \mathbb{E}_{q(w,b)}\!\bigl[\ln p(D \mid w,b)\;+\;\ln p(w,b)\;-\;\ln q(w,b)\bigr].
\]

Maximizing the ELBO w.r.t. \(\{\mu_w, s_w, \mu_b, s_b\}\) pushes \(q\) to be close to the true posterior \(p(w,b \mid D)\).

---

# 3. A Single Iteration “by Hand”

We’ll do **one iteration** of “stochastic VI”:  
1. **Sample** \(w'\) and \(b'\) from the current \(q\).  
2. **Evaluate** the log-likelihood and log-prior at \((w', b')\).  
3. **Compute** the gradient of the ELBO estimate w.r.t. \(\mu_w, s_w, \dots\).  
4. **Update** the parameters.

Let’s see how it might look numerically, in a *highly abbreviated* form.

> **Note**: This is not a “complete” derivation or a real optimization loop. It’s just an illustration of how one step’s numbers could pan out.

### 3.1 Step 1: Sample \((w', b')\)

- Current \((\mu_w, s_w) = (0, 1)\).
- Current \((\mu_b, s_b) = (0, 1)\).

Hence,
\[
w' \sim \mathcal{N}(0, 1^2) \quad\longrightarrow\quad \text{(say we draw) } w' = +0.37.
\]
\[
b' \sim \mathcal{N}(0, 1^2) \quad\longrightarrow\quad \text{(say we draw) } b' = -0.11.
\]

### 3.2 Step 2: Compute Log Terms

We estimate the **ELBO** by approximating the expectation with a single sample:

\[
\widehat{\text{ELBO}} 
\approx \ln p(D\mid w',b') + \ln p(w',b') - \ln q(w',b').
\]

Let’s break those down.

---

#### 3.2.1 Log-likelihood: \(\ln p(D\mid w',b')\)

We have **two data points** \((x_1=1,y_1=1.8)\) and \((x_2=3,y_2=2.7)\). Summation of log-likelihood:

\[
\ln p(D\mid w',b') 
= \sum_{i=1}^{2}\ln \Bigl[\mathcal{N}\bigl(y_i \mid w'\,x_i+b',\;0.2^2\bigr)\Bigr].
\]

- For \(i=1\):
  \[
  \hat{y_1} = w' \cdot x_1 + b' = 0.37 \cdot 1 + (-0.11) = 0.26.
  \]
  \[
  \text{residual}_1 = y_1 - \hat{y_1} = 1.8 - 0.26 = 1.54.
  \]
  \[
  \ln p_1 = \ln \Bigl[\frac{1}{\sqrt{2\pi}\cdot 0.2} \exp\bigl(-\tfrac{(1.54)^2}{2\cdot0.2^2}\bigr)\Bigr].
  \]
  Numerically:
  \[
   \frac{(1.54)^2}{2\cdot0.2^2} = \frac{2.3716}{0.08} \approx 29.645.
  \]
  That exponential is extremely small! So \(\ln p_1\) is quite negative. Roughly:
  \[
  \ln p_1 \approx \ln \Bigl[\frac{1}{0.5} \times e^{-29.645}\Bigr] 
           = \ln(2) + (-29.645) 
           \approx 0.693 - 29.645 
           \approx -28.95.
  \]

- For \(i=2\):
  \[
  \hat{y_2} = w' \cdot x_2 + b' = 0.37 \cdot 3 + (-0.11) = 1.0.
  \]
  \[
  \text{residual}_2 = y_2 - \hat{y_2} = 2.7 - 1.0 = 1.7.
  \]
  \[
  \ln p_2 \approx \ln\Bigl[\frac{1}{0.5} e^{-\frac{(1.7)^2}{0.08}}\Bigr].
  \]
  \[
  \frac{(1.7)^2}{0.08} = \frac{2.89}{0.08} = 36.125.
  \]
  So
  \[
  \ln p_2 \approx \ln(2) - 36.125 
           \approx 0.693 - 36.125 
           \approx -35.432.
  \]

Therefore,
\[
\ln p(D\mid w',b') 
\approx -28.95 + (-35.43) 
= -64.38.
\]

It’s a very poor fit to the data if \((w'=0.37, b'=-0.11)\) because our residuals are large given the small noise \(\sigma=0.2\).

---

#### 3.2.2 Log-prior: \(\ln p(w',b')\)

\[
\ln p(w',b') = \ln p(w') + \ln p(b'), 
\]
with
\[
p(w') = \mathcal{N}(w' \mid 0, 1^2), \quad p(b') = \mathcal{N}(b' \mid 0, 1^2).
\]

- \( w' = 0.37 \implies \ln p(w') = \ln\Bigl[\tfrac{1}{\sqrt{2\pi}} e^{-0.37^2/2}\Bigr]. \)  
  Numerically, \(\frac{(0.37)^2}{2} = 0.06845\). Then:
  \[
  \ln p(w') \approx \ln\Bigl(\frac{1}{2.5066}\Bigr) - 0.06845
               = -0.92 - 0.06845 
               \approx -0.988.
  \]
- \( b' = -0.11 \implies \ln p(b') \approx \ln\Bigl(\frac{1}{2.5066}\Bigr) - \frac{(0.11)^2}{2}. \)  
  \(\frac{(0.11)^2}{2} = 0.00605\). So
  \[
  \ln p(b') \approx -0.92 - 0.00605 = -0.926.
  \]
Thus,
\[
\ln p(w',b') \approx -0.988 + (-0.926) = -1.914.
\]

---

#### 3.2.3 Log of the Variational Distribution: \(\ln q(w',b')\)

\[
\ln q(w',b') = \ln q(w') + \ln q(b').
\]

Given the current parameters \(\mu_w=0, s_w=1\) and \(\mu_b=0, s_b=1\):

- \(\ln q(w') = \ln \mathcal{N}(w'=0.37 \mid 0, 1)\), basically the same formula as the prior with mean=0 and std=1. So:
  \[
  \ln q(w') \approx -0.988 \quad (\text{same as } \ln p(w')).
  \]
- \(\ln q(b') \approx -0.926 \quad (\text{same as } \ln p(b')).\)

Hence,
\[
\ln q(w',b') \approx -1.914.
\]

---

### 3.3 Combine for the Single-Sample ELBO

\[
\widehat{\text{ELBO}}
= \underbrace{\ln p(D\mid w',b')}_{-64.38} 
+ \underbrace{\ln p(w',b')}_{-1.914} 
- \underbrace{\ln q(w',b')}_{-1.914}
\approx -64.38 + (-1.914) - (-1.914).
\]
Notice that the \(\ln p(w',b')\) and \(\ln q(w',b')\) are equal in magnitude but opposite sign if \(p\) and \(q\) share mean=0, stdev=1 for that sample. This means they **cancel out** in this step:

\[
= -64.38 + (-1.914) + 1.914 \approx -64.38.
\]

So the single-sample estimate of the ELBO is about **\(-64.38\)** for that particular draw \((w'=0.37, b'=-0.11)\).

### 3.4 Gradient and Parameter Update (Conceptual)

In practice, **Pyro, TensorFlow Probability, or your own code** would do:

1. **Differentiate** the sampled loss = \(-\widehat{\text{ELBO}}\) w.r.t. \(\mu_w, s_w, \mu_b, s_b\).  
2. **Update** the parameters in the direction that **increases** the ELBO.

Because the likelihood is so small at \((w'=0.37, b'=-0.11)\) for these data, the gradient would push \(\mu_w\) and \(\mu_b\) toward something that yields bigger likelihood. Over many iterations, \(\mu_w\) will approach a more plausible slope, and \(\mu_b\) will approach a more plausible intercept.

---

# 4. Interpreting What’s Going On

1. **Prior**: We start believing \(w\approx 0\), \(b\approx 0\).  
2. **Likelihood**: The data \((1,1.8)\) and \((3,2.7)\) suggests a slope and intercept that make \(y\approx 1.8\) at \(x=1\), and \(y\approx 2.7\) at \(x=3\). Roughly, you might guess \(w\approx 0.45\) or \(0.5\) and \(b\approx 1.35\). (This is naive eyeballing; the actual best fit depends on the noise assumptions.)  
3. **Posterior**: Over multiple draws and multiple gradient steps, the approximate posterior \(q(w,b)\) will converge so that \(\mu_w,\mu_b\) center near the best-fit line, and \(s_w^2,s_b^2\) (the variances) reflect our *uncertainty*.

Thus, by the end of training, a BNN yields a **distribution** over \((w,b)\), not just a single best-fit pair. This distribution is “peakier” if the data strongly constrain the slope/intercept, or “wider” if the data is noisy or ambiguous.

---

# 5. Putting It All Together

- **In Standard Neural Nets**: We find a single \(w^*, b^*\) that *best fits* the data.  
- **In Bayesian Neural Nets**: We maintain **a distribution** \(p(w,b\mid D)\). Because we only approximate it with \(q(w,b)\), we do a gradient-based procedure that tries to push \(q\) to match the “true” posterior distribution.  

**The big difference** is that from a BNN, we can sample many \((w, b)\) pairs from \(q\) and get a **distribution of predictions**. That distribution can tell us **how uncertain** the model is about the slope or intercept, and thereby about predictions for new \(x\).

---

## Final Thoughts

1. **This example** used a toy “network” with a single weight and bias to keep the math tractable.  
2. **Real BNNs** have many layers and thousands (or millions) of weights. You’d do the same logic but let a library handle the sampling and gradient-based updates.  
3. **Despite the simplicity**, the principle is the same: we define a prior, define a likelihood, form a posterior, and do approximate inference.  
4. **In practice**, you’d run **thousands of iterations** of the “sample \(\to\) compute \(\to\) gradient \(\to\) update” loop until the ELBO converges, not just one iteration.

This completes the **“by hand”** demonstration for a **Bayesian Neural Network** with two data points!