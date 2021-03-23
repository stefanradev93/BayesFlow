# BayesFlow
Welcome to the beta-version of our BayesFlow library for simulation-based Bayesian parameter estimation and model comparison!

A cornerstone idea of amortized Bayesian inference is to employ generative neural networks for parameter estimation, model comparison and model validation.
when working with intractable simulators whose behavior as a whole is too complex to be described analytically. The figure below presents a higher-level overview of this idea. 

![Overview](https://github.com/stefanradev93/BayesFlow/blob/master/img/high_level_framework.png)

A short conference paper reviewing amortized inference with a focus on cognitive modeling can be found here:

https://arxiv.org/abs/2005.03899

## Parameter estimation

The algorithm for parameter estimation is based on our paper:

Radev, S. T., Mertens, U. K., Voss, A., Ardizzone, L., & Köthe, U. (2020). BayesFlow: Learning complex stochastic models with invertible neural networks. <em>IEEE Transactions on Neural Networks and Learning Systems</em>, available for free at:

https://arxiv.org/abs/2003.06281

Currently, the following training approaches are implemented:
1. Online training
2. Offline training (external simulations)
3. Offline training (internal simulations)
4. Experience replay
5. Round-based training

## Model comparison

The algorithm for model comparison is based on our paper:

Radev, S. T., D'Alessandro, M., Bürkner, P. C., Mertens, U. K., Voss, A., & Köthe, U. (2020). Amortized bayesian model comparison with evidential deep learning. <em>arXiv preprint arXiv:2004.10629</em>, available for free at:

https://arxiv.org/abs/2004.10629
