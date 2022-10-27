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

![Overview](https://github.com/stefanradev93/BayesFlow/blob/9308cc044b28fc0d7d02714dd20dc9b206fa040b/img/high_level_framework.png?raw=true)

Currently, the following training approaches are implemented:
1. Online training
2. Offline training (external simulations)
3. Offline training (internal simulations)
4. Experience replay
5. Round-based training

## Parameter Estimation

The BayesFlow approach for amortized parameter estimation is based on our paper:

Radev, S. T., Mertens, U. K., Voss, A., Ardizzone, L., & Köthe, U. (2020). BayesFlow: Learning complex stochastic models with invertible neural networks. <em>IEEE Transactions on Neural Networks and Learning Systems</em>, available for free at: https://arxiv.org/abs/2003.06281. The general pattern for building amortized posterior approximators is illsutrated below:

![BayesFlow](https://github.com/stefanradev93/BayesFlow/blob/Future/docs/source/tutorial_notebooks/img/trainer_connection.png?raw=true)

### Minimal Example

For instance, in order to tackle a simple memoryless model with 10 free parameters, we first need to set up an optional summary network and an inference network:
```python
# Use default settings
summary_net = InvariantNetwork()
inference_net = InvertibleNetwork({'n_params': 10})
# Connect summary and inference network
amortizer = AmortizedPosterior(inference_net, summary_net)
```
Next, we define a generative model which connects a *prior* (a function returning random draws from the prior distribution over parameters) with a *simulator* (a function accepting the prior draws as arguments) and returning a simulated data set with *n_obs* potentially multivariate observations.
```python
generative_model = GenerativeModel(prior, simulator)
```
Finally, we connect the networks with the generative model via a trainer instance:
```python
# Using default settings
trainer = Trainer(
    network=amortizer, 
    generative_model=generative_model
)
```
We are now ready to train an amortized posterior approximator. For instance, to run online training, we simply call
```python
losses = trainer.train_online(epochs=50, iterations_per_epoch=1000, batch_size=64)
```
which performs online training for 50 epochs of 1000 iterations (batch simulations with 64 simulations per batch). See the [tutorial notebooks](docs/source/tutorial_notebooks) for more examples. Posterior inference is then fast and easy:
```python
# Obtain 5000 samples from the posterior given obs_data
samples = amortizer.sample(obs_data, n_samples=5000)
```
### Further Reading

Coming soon...

## Model Misspecification

Coming soon...

## Model Comparison

Coming soon...

## Likelihood emulation

Coming soon...
