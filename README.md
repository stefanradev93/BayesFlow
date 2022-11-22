# BayesFlow

Welcome to the beta-version of our BayesFlow library for simulation-based Bayesian workflows.

For starters, check out the walk-through notebooks:
1. [Basic amortized posterior estimation](docs/source/tutorial_notebooks/Intro_Amortized_Posterior_Estimation.ipynb) 
2. [Intermediate posterior estimation](docs/source/tutorial_notebooks/Covid19_Initial_Posterior_Estimation.ipynb) 
3. [Posterior estimation for ODEs](docs/source/tutorial_notebooks/Linear%20ODE%20system.ipynb)
4. Coming soon...

## Project Documentation
The project documentation is available at <http://bayesflow.readthedocs.io>

## Conceptual Overview

A cornerstone idea of amortized Bayesian inference is to employ generative neural networks for parameter estimation, model comparison, and model validation
when working with intractable simulators whose behavior as a whole is too complex to be described analytically. The figure below presents a higher-level overview of neurally bootstrapped Bayesian inference. 

![Overview](https://github.com/stefanradev93/BayesFlow/blob/Future/img/high_level_framework.png?raw=true)

## Parameter Estimation

The BayesFlow approach for amortized parameter estimation is based on our paper:

Radev, S. T., Mertens, U. K., Voss, A., Ardizzone, L., & Köthe, U. (2020). BayesFlow: Learning complex stochastic models with invertible neural networks. <em>IEEE Transactions on Neural Networks and Learning Systems</em>, available for free at: https://arxiv.org/abs/2003.06281. 

### Minimal Example

```python
import numpy as np
import bayesflow as bf

# First, we define a simple 2D toy model with a Gaussian prior and a Gaussian simulator (likelihood):
def prior(D=2, mu=0., sigma=1.0):
    return np.random.default_rng().normal(loc=mu, scale=sigma, size=D)

def simulator(theta, n_obs=50, scale=1.0):
    return np.random.default_rng().normal(loc=theta, scale=scale, size=(n_obs, theta.shape[0]))

# Then, we create our BayesFlow setup consisting of a summary and an inference network:
summary_net = bf.networks.InvariantNetwork()
inference_net = bf.networks.InvertibleNetwork(num_params=2)
amortizer = bf.amortizers.AmortizedPosterior(inference_net, summary_net)

# Next, we connect the `prior` with the `simulator` using a `GenerativeModel` wrapper:
generative_model = bf.simulation.GenerativeModel(prior, simulator)

# Finally, we connect the networks with the generative model via a `Trainer` instance:
trainer = bf.trainers.Trainer(amortizer=amortizer, generative_model=generative_model)

# We are now ready to train an amortized posterior approximator. For instance, to run online training, we simply call:
losses = trainer.train_online(epochs=10, iterations_per_epoch=500, batch_size=32)
```

Before inference, we can use simulation-based calibration (SBC, https://arxiv.org/abs/1804.06788) to check the computational faithfulness of the model-amortizer combination:
```python
fig = trainer.diagnose_sbc_histograms()
```
![SBC](https://github.com/stefanradev93/BayesFlow/blob/Future/img/showcase_sbc.png?raw=true)
Amortized inference on new (real or simulated) data is then easy and fast:
```python
# Simulate 200 new data sets and generate 500 posterior draws per data set
new_sims = trainer.configurator(generative_model(200))
posterior_draws = amortizer.sample(new_sims, n_samples=500)
```
We can then quickly inspect the parameter recoverability of the model
```python
fig = bf.diagnostics.plot_recovery(posterior_draws, new_sims['parameters'])
```
![Recovery](https://github.com/stefanradev93/BayesFlow/blob/Future/img/showcase_recovery.png?raw=true)
or look at single posteriors in relation to the prior:
```python
fig = bf.diagnostics.plot_posterior_2d(posterior_draws[0], prior=generative_model.prior)
```
![Posterior](https://github.com/stefanradev93/BayesFlow/blob/Future/img/showcase_posterior.png?raw=true)

### Further Reading

Coming soon...

## Model Misspecification

What if we are dealing with misspecified models? That is, how faithful is our amortized inference if the generative model is a poor representation of reality? A modified loss function optimizes the learned summary statistics towards a unit Gaussian and reliably detects model misspecification during inference time.

![Model Misspecification](https://github.com/stefanradev93/BayesFlow/blob/Future/docs/source/images/model_misspecification_amortized_sbi.png?raw=true)




## Model Comparison

Coming soon...

## Likelihood emulation

Coming soon...
