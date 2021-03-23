# BayesFlow
Welcome to the beta-version of our BayesFlow library for simulation-based Bayesian parameter estimation and model comparison!

A cornerstone idea of amortized Bayesian inference is to employ generative neural networks for parameter estimation, model comparison and model validation.
when working with intractable simulators whose behavior as a whole is too complex to be described analytically. The figure below presents a higher-level overview of this idea. 

![Overview](https://github.com/stefanradev93/BayesFlow/blob/master/img/high_level_framework.png?s=200)

## Parameter estimation

The algorithm for parameter estimation is based on our paper:

Radev, S. T., Mertens, U. K., Voss, A., Ardizzone, L., & Köthe, U. (2020). BayesFlow: Learning complex stochastic models with invertible neural networks. <em>IEEE Transactions on Neural Networks and Learning Systems</em>.
