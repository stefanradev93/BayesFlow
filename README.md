# BayesFlow
Welcome to the beta-version of our BayesFlow library for simulation-based Bayesian parameter estimation and model comparison!

For starters, check out the walkthrough notebooks *Parameter_Estimation_Workflow.ipynb* and *Model_Comparison_Workflow.ipynb*. For short code samples, please read below!

## Conceptual Overview

A cornerstone idea of amortized Bayesian inference is to employ generative neural networks for parameter estimation, model comparison and model validation
when working with intractable simulators whose behavior as a whole is too complex to be described analytically. The figure below presents a higher-level overview of neurally bootstrapped Bayesian inference. 

![Overview](https://github.com/stefanradev93/BayesFlow/blob/master/img/high_level_framework.png)

A short conference paper reviewing amortized Bayesian inference with a focus on cognitive modeling can be found here:

https://arxiv.org/abs/2005.03899

## Parameter Estimation

The BayesFlow approach for parameter estimation incorporates a *summary network* and an *inference network* which are jointly optimized to invert a complex computational model (simulator). The summary network is responsible for learning the most informative data representations (i.e., summary statistics) in an end-to-end manner. The inference network is responsible for learning an invertible mapping between the posterior and an easy-to-sample-from latent space (e.g., Gaussian) for *any* possible observation or set of observations arising from the simulator. The BayesFlow method for amortized parameter estimation is based on our paper:

Radev, S. T., Mertens, U. K., Voss, A., Ardizzone, L., & Köthe, U. (2020). BayesFlow: Learning complex stochastic models with invertible neural networks. <em>IEEE Transactions on Neural Networks and Learning Systems</em>, available for free at:

https://arxiv.org/abs/2003.06281

The general workflow (training and inference phase) with BayesFlow is illustrated below.

![BayesFlow](https://github.com/stefanradev93/BayesFlow/blob/master/img/BayesFlow.png)

Currently, the following training approaches are implemented:
1. Online training
2. Offline training (external simulations)
3. Offline training (internal simulations)
4. Experience replay
5. Round-based training

In order to ensure algorithmic alignment between the neural approximator and the computational model (simulator), we recommend the following neural architectural considerations:

### Stateless (memoryless) models
Stateless models typically generate IID observations, which imply exchangeability and induce permutation invariant posteriors. In other words, changing (permuting) the order of individual elements should not change the associated likelihood or posterior. An example BayesFlow architecture for tackling stateless models is depicted below.

![Stateless](https://github.com/stefanradev93/BayesFlow/blob/master/img/Stateless_Models.png)

You can read more about designing invariant networks in the excellent paper by Benjamin Bloem-Reddy and Yee Whye Teh, available at https://arxiv.org/abs/1901.06082.

For instance, in order to tackle a memoryless model with 10 free parameters via BayesFlow, we first need to set-up the summary and inference networks:
```python
# Use default settings
summary_net = InvariantNetwork()
inference_net = InvertibleNetwork({'n_params': 10})
# Connect summary and inference network
amortizer = SingleModelAmortizer(inference_net, summary_net)
```
Next, we define a generative model which connects a *prior* (a function returning random draws from the prior distribution over parameters) with a *simulator* (a function accepting the prior draws as arguments) and returning a simulated data set with *n_obs*) potentially multivariate observations.
```python
generative_model = GenerativeModel(prior, simulator)
```
Finally, we connect the networks with the generative model via a trainer instance:
```python
# Using default settings
trainer = ParameterEstimationTrainer(
    network=amortizer, 
    generative_model=generative_model
)
```
We are now ready to train an amortized parameter estimator via various options. For instance, to run online training, we simply call
```python
losses = trainer.train_online(epochs=50, iterations_per_epoch=1000, batch_size=64, n_obs=200)
```
which performs online training for 50 epochs of 1000 iterations (batch simulations with 64 simulations per batch). The shape of each batch is thus (64, 200, summary_dim), corresponding to 64 simulations per batch, 200 observations per simulated data set, and *summary_dim* output dimensions of the final layer of the permutation-invariant summary network. See the *Parameter_Estimation_Workflow.ipynb* notebook for a detailed walkthrough. 

Posterior inference is then fast and easy:
```python
# Obtain 5000 samples from the posterior given obs_data
samples = amortizer.sample(obs_data, n_samples=5000)
```

### Stateful models
Stateful models incorporate some form of memory and are thus capable of generating observations with complex dependencies (i.e., non-IID). A prime example are dynamic models, which typically describe the evolution trajectory of a system or a process, such as an infectious disease, over time. Observations generated from such models are usually the solution of a stochastic differential equation(SDE) or time-series and thus imply a more complex probabilistic symmetry than those generated from memoryless models. An example BayesFlow architecture for tackling stateful models is depicted below.

![Stateful](https://github.com/stefanradev93/BayesFlow/blob/master/img/Stateful_Models.png)

We used the above architecture for modeling the early Covid-19 outbreak in Germany: https://arxiv.org/abs/2010.00300.

### Joint models
Joint models present an attempt to account for different processes (e.g., neural and cognitive) within a single composite model. Thus, joint models integrate different sources and types of data and require morec omplex summary architectures. An example BayesFlow architecture for three hypothetical data sources is depicted below.

![Joint](https://github.com/stefanradev93/BayesFlow/blob/master/img/Joint_Models.png)

## Model Comparison

The algorithm for model comparison is based on our paper:

Radev, S. T., D'Alessandro, M., Mertens, U. K., Voss, A., & Köthe, U.,Bürkner, P. C. (2020). Amortized bayesian model comparison with evidential deep learning. <em>arXiv preprint arXiv:2004.10629</em>, available for free at:

https://arxiv.org/abs/2004.10629
