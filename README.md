# BayesFlow <img src="img/bayesflow_hex.png" align="right" width=20% height=20% />

[![Actions Status](https://github.com/stefanradev93/bayesflow/workflows/Tests/badge.svg)](https://github.com/stefanradev93/bayesflow/actions)
[![Licence](https://img.shields.io/github/license/stefanradev93/BayesFlow)](https://img.shields.io/github/license/stefanradev93/BayesFlow)

Welcome to our BayesFlow library for efficient simulation-based Bayesian workflows! Our library enables users to create specialized neural networks for *amortized Bayesian inference*, which repays users with rapid statistical inference after a potentially longer simulation-based training phase.

For starters, check out some of our walk-through notebooks:

1. [Quickstart amortized posterior estimation](docs/source/tutorial_notebooks/Intro_Amortized_Posterior_Estimation.ipynb)
2. [Principled Bayesian workflow for cognitive models](docs/source/tutorial_notebooks/LCA_Model_Posterior_Estimation.ipynb)
3. [Posterior estimation for ODEs](docs/source/tutorial_notebooks/Linear_ODE_system.ipynb)
4. [Posterior estimation for SIR-like models](docs/source/tutorial_notebooks/Covid19_Initial_Posterior_Estimation.ipynb)

## Project Documentation

The project documentation is available at <https://bayesflow.readthedocs.io>

## Installation

See [INSTALL.rst](INSTALL.rst) for installation instructions.

## Conceptual Overview

A cornerstone idea of amortized Bayesian inference is to employ generative
neural networks for parameter estimation, model comparison, and model validation
when working with intractable simulators whose behavior as a whole is too
complex to be described analytically. The figure below presents a higher-level
overview of neurally bootstrapped Bayesian inference.

<img src="img/high_level_framework.png" width=80% height=80%>

## Getting Started: Parameter Estimation

The core functionality of BayesFlow is amortized Bayesian posterior estimation, as described in our paper:

Radev, S. T., Mertens, U. K., Voss, A., Ardizzone, L., & Köthe, U. (2020).
BayesFlow: Learning complex stochastic models with invertible neural networks.
<em>IEEE Transactions on Neural Networks and Learning Systems</em>, available
for free at: https://arxiv.org/abs/2003.06281.

However, since then, we have substantially extended the BayesFlow library such that
it is now much more general and cleaner than what we describe in the above paper.

### Minimal Example

```python
import numpy as np
import bayesflow as bf
```

To introduce you to the basic workflow of the library, let's consider
a simple 2D Gaussian model, from which we want to obtain
posterior inference.  We assume a Gaussian simulator (likelihood)
and a Gaussian prior for the means of the two components,
which are our only model parameters in this example:

```python
def simulator(theta, n_obs=50, scale=1.0):
    return np.random.default_rng().normal(loc=theta, scale=scale, size=(n_obs, theta.shape[0]))

def prior(D=2, mu=0., sigma=1.0):
    return np.random.default_rng().normal(loc=mu, scale=sigma, size=D)
```

Then, we connect the `prior` with the `simulator` using a `GenerativeModel` wrapper:

```python
generative_model = bf.simulation.GenerativeModel(prior, simulator)
```

Next, we create our BayesFlow setup consisting of a summary and an inference network:

```python
summary_net = bf.networks.InvariantNetwork()
inference_net = bf.networks.InvertibleNetwork(num_params=2)
amortizer = bf.amortizers.AmortizedPosterior(inference_net, summary_net)
```

Finally, we connect the networks with the generative model via a `Trainer` instance:

```python
trainer = bf.trainers.Trainer(amortizer=amortizer, generative_model=generative_model)
```

We are now ready to train an amortized posterior approximator. For instance,
to run online training, we simply call:

```python
losses = trainer.train_online(epochs=10, iterations_per_epoch=500, batch_size=32)
```

Before inference, we can use simulation-based calibration (SBC,
https://arxiv.org/abs/1804.06788) to check the computational faithfulness of
the model-amortizer combination:

```python
fig = trainer.diagnose_sbc_histograms()
```

<img src="img/showcase_sbc.png" width=65% height=65%>

The histograms are roughly uniform and lie within the expected range for
well-calibrated inference algorithms as indicated by the shaded gray areas.
Accordingly, our amortizer seems to have converged to the intended target.

Amortized inference on new (real or simulated) data is then easy and fast.
For example, we can simulate 200 new data sets and generate 500 posterior draws
per data set:

```python
new_sims = trainer.configurator(generative_model(200))
posterior_draws = amortizer.sample(new_sims, n_samples=500)
```

We can then quickly inspect the how well the model can recover its parameters
across the simulated data sets.

```python
fig = bf.diagnostics.plot_recovery(posterior_draws, new_sims['parameters'])
```

<img src="img/showcase_recovery.png" width=65% height=65%>

For any individual data set, we can also compare the parameters' posteriors with
their corresponding priors:

```python
fig = bf.diagnostics.plot_posterior_2d(posterior_draws[0], prior=generative_model.prior)
```

<img src="img/showcase_posterior.png" width=45% height=45%>

We see clearly how the posterior shrinks relative to the prior for both
model parameters as a result of conditioning on the data.

### References and Further Reading

- Radev, S. T., Mertens, U. K., Voss, A., Ardizzone, L., & Köthe, U. (2020).
BayesFlow: Learning complex stochastic models with invertible neural networks.
<em>IEEE Transactions on Neural Networks and Learning Systems</em>, available
for free at: https://arxiv.org/abs/2003.06281.

- Radev, S. T., Graw, F., Chen, S., Mutters, N. T., Eichel, V. M., Bärnighausen, T., & Köthe, U. (2021).
OutbreakFlow: Model-based Bayesian inference of disease outbreak dynamics with invertible neural networks and its application to the COVID-19 pandemics in Germany. <em>PLoS computational biology</em>, 17(10), e1009472.

- Bieringer, S., Butter, A., Heimel, T., Höche, S., Köthe, U., Plehn, T., & Radev, S. T. (2021).
Measuring QCD splittings with invertible networks. <em>SciPost Physics</em>, 10(6), 126.

- von Krause, M., Radev, S. T., & Voss, A. (2022).
Mental speed is high until age 60 as revealed by analysis of over a million participants.
<em>Nature Human Behaviour</em>, 6(5), 700-708.

## Model Misspecification

What if we are dealing with misspecified models? That is, how faithful is our
amortized inference if the generative model is a poor representation of reality?
A modified loss function optimizes the learned summary statistics towards a unit
Gaussian and reliably detects model misspecification during inference time.

![](docs/source/images/model_misspecification_amortized_sbi.png?raw=true)

In order to use this method, you should only provide the `summary_loss_fun` argument
to the `AmortizedPosterior` instance:

```python
amortizer = bf.amortizers.AmortizedPosterior(inference_net, summary_net, summary_loss_fun='MMD')
```

The amortizer knows how to combine its losses.

### References and Further Reading

- Schmitt, M., Bürkner P. C., Köthe U., & Radev S. T. (2022). Detecting Model
Misspecification in Amortized Bayesian Inference with Neural Networks. <em>ArXiv
preprint</em>, available for free at: https://arxiv.org/abs/2112.08866

## Model Comparison

Example coming soon...

### References and Further Reading

- Radev S. T., D’Alessandro M., Mertens U. K., Voss A., Köthe U., & Bürkner P.
C. (2021). Amortized Bayesian Model Comparison with Evidental Deep Learning.
<em>IEEE Transactions on Neural Networks and Learning Systems</em>.
doi:10.1109/TNNLS.2021.3124052 available for free at: https://arxiv.org/abs/2004.10629

- Schmitt, M., Radev, S. T., & Bürkner, P. C. (2022). Meta-Uncertainty in
Bayesian Model Comparison. <em>ArXiv preprint</em>, available for free at:
https://arxiv.org/abs/2210.07278

## Likelihood emulation

Example coming soon...
