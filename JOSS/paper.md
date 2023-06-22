---
title: "BayesFlow: Amortized Bayesian Workflows With Neural Networks"
tags:
  - "simulation-based inference"
  - "likelihood-free inference"
  - Bayesian inference
  - Python
date: "05 June 2023"
output:
  html_document:
  df_print: paged
authors:
  - name: Stefan T. Radev
    orcid: "0000-0002-6702-9559"
    corresponding: true
    equal-contrib: true
    affiliation: 1
  - name: Marvin Schmitt
    orcid: "0000-0002-7967-4723"
    equal-contrib: true
    affiliation: 2
  - name: Valentin Pratz
    affiliation: 3
  - name: Yannik Schälte
    affiliation: 4
    orcid: "0000-0003-1293-820X"
  - name: Lukas Schumacher
    affiliation: 5
    orcid: "0000-0003-1512-8288"
  - name: Lasse Elsemüller
    affiliation: 5
    orcid: "0000-0003-0368-720X"
  - name: Ullrich Köthe
    affiliation: 3
    orcid: "0000-0001-6036-1287"
    equal-contrib: true
  - name: Paul-Christian Bürkner
    orcid: "0000-0001-5765-8995"
    affiliation: "2,6"
    equal-contrib: true
bibliography: paper.bib
affiliations:
  - name: Cluster of Excellence STRUCTURES, Heidelberg University, Germany
    index: 1
  - name: Cluster of Excellence SimTech, University of Stuttgart, Germany
    index: 2
  - name: Visual Learning Lab, Heidelberg University, Germany
    index: 3
  - name: Life and Medical Sciences (LIMES) Institute, University of Bonn, Germany
    index: 4
  - name: Institute for Psychology, Heidelberg University, Germany
    index: 5
  - name: Department of Statistics, TU Dortmund University, Germany
    index: 6
---

# Summary
Modern Bayesian inference involves a mixture of computational techniques for estimating, validating, and drawing conclusions from probabilistic models as part of principled workflows for data analysis [@burkner_models_2022; @gelman_bayesian_2020; @schad2021toward].  Typical problems in Bayesian workflows are the approximation of intractable posterior distributions for diverse model types and the comparison of competing models of the same process in terms of their predictive or generative performance. However, despite their theoretical appeal and utility, the practical execution of Bayesian workflows is often limited by computational bottlenecks: Obtaining even a single posterior may already take a long time, such that repeated estimation for the purpose of model validation or calibration becomes completely unfeasible.

Our `BayesFlow` software brings *amortized Bayesian inference* (ABI) into the scope of Bayesian workflows by enabling users to train custom-tailored neural networks on model simulations and re-use these networks for any subsequent application of the model. Thus, ABI unlocks the potential of powerful tools for estimation, validation, and comparison of complex models (e.g., models with intractable likelihoods or time-varying parameters) that are often out of reach for standard methods. For instance, testing a model's ability to recover its parameters [@schad2021toward] or validating computational fidelity via calibration methods [@talts2018; @sailynoja2022graphical] may necessitate the estimation of thousands of posterior distributions. This can take days on a compute cluster, but is nearly instantaneous in the context of ABI.

To this end, `BayesFlow` incorporates *simulation-based* training of established neural network architectures, such as transformers [@vaswani2017attention] and normalizing flows [@rezende2015normalizing; @papamakarios2021normalizing] for data compression and inference. The guiding motif behind the design of `BayesFlow` is to abstract away technical details that are not necessarily relevant for practical applications, while providing robust default settings that work well across applications and require minimal need for manual tuning by the user. At the same time, `BayesFlow` implements a modular software architecture, allowing machine learning scientists to modify every component of the pipeline for cutting-edge academic research at the frontier of simulation-based inference.

# Statement of Need

`BayesFlow` features functionality specifically designed for building and validating amortized Bayesian workflows with the help of neural networks. It is built on top of `TensorFlow` [@abadi2016tensorflow] and thereby enables off-the-shelf support for GPU and TPU acceleration. Furthermore, it can seamlessly interact with TensorFlow Probability [@dillon2017tensorflow] for flexible latent distributions and a variety of joint priors.

![`BayesFlow` defines a formal workflow for data generation, neural approximation, and model criticism.\label{fig:figure1}](https://hackmd.io/_uploads/rJXliqZO2.png)

\autoref{fig:figure1} outlines a typical workflow in the context of amortized posterior and likelihood estimation. A simulator coupled with a prior defines a generative Bayesian model which may depend on various (optional) context variates (e.g., varying numbers of observations, design matrices, positional encodings, etc.). The generative scope of the model, together with the range of context variables, determine the *scope of amortization*, that is, over which types of data the neural approximator(s) can be applied without re-training. The neural approximators interact with model outputs (parameters, data) and context variates through a configurator, which is responsible for carrying out transformations (e.g., input normalization, double-to-float conversion, etc.) that are not part of the model but may facilitate neural network training and convergence. 

\autoref{fig:figure1} also illustrates an example configuration of three neural approximators: 1) a summary network to compress individual data points, sets, or time series into informative embeddings; 2) a posterior network to learn an amortized approximate posterior; and 3) a likelihood network to learn an amortized approximate likelihood. Along with their inputs, \autoref{fig:figure1} depicts the standalone and joint capabilities of the networks when applied in isolation or in tandem, respectively.

Currently, the software features four key capabilities for enhancing Bayesian workflows, which have been described in the referenced works:

1. **Amortized posterior estimation:** Train a generative network to efficiently infer full posteriors (i.e., solve the inverse problem) for all existing and future data compatible with a simulation model [@radev2020bayesflow]. 
2. **Amortized likelihood estimation:** Train a generative network to efficiently emulate a simulation model (i.e., solve the forward problem) for all possible parameter configurations or interact with external probabilistic programs [@radev2023jana; @boelts2022flexible].
3. **Amortized model comparison:** Train a neural classifier to recognize the "best" model in a set of competing candidates [@radev2020evidential; @schmitt2022meta; @elsemuller2023deep] or combine amortized posterior and likelihood estimation to compute Bayesian evidence and out-of-sample predictive performance [@radev2023jana].
4. **Model misspecification detection:** Ensure that the resulting posteriors are faithful approximations of the otherwise intractable target posterior, even when simulations do not perfectly represent reality [@schmitt2021detecting; @radev2023jana].


`BayesFlow` has been used for amortized Bayesian inference in various areas of applied research, such as epidemiology [@radev2021outbreakflow], cognitive modeling [@von2022mental,wieschen2020jumping,sokratous2023ask], computational psychiatry [@d2020bayesian], neuroscience [@ghaderi2022general], particle physics [@bieringer2021measuring], agent-based econometrics models [@shiono2021estimation], seismic imaging [@siahkoohi2023reliable], user behavior [@moon2023amortized], structural health monitoring [@zeng2023probabilistic], aerospace [@tsilifis2022inverse] and wind turbine design [@noever2022model], micro-electro-mechanical systems testing [@heringhaus2022towards], and fractional Brownian motion [@verdier2022variational].


# Related Software
When a non-amortized inference procedure does not create a computational bottleneck, approximate Bayesian computation (ABC) might be an appropriate tool. This is the case when a single data set needs to be analyzed, when an infrastructure for parallel computing is readily available, or when repeated re-fits of a model (e.g., cross validation) are not desired.
A variety of mature Python packages for ABC exist, such as PyMC [@Salvatier2016], pyABC [@schaelte2022pyabc], or ELFI [@lintusaari2018elfi]. In contrast to these packages, `BayesFlow` focuses on amortized inference, but can also interact with ABC samplers (e.g., using pre-trained summary networks for learned summary statistics in ABC).

When it comes to amortized inference with neural networks, the `sbi` toolkit enables both likelihood and posterior estimation using different inference algorithms, such as Sequential Neural Posterior Estimation [@greenberg2019automatic] and Sequential Neural Likelihood Estimation [@papamakarios2019sequential]. The `Swyft` library focuses on Bayesian parameter inference in physics and astronomy. `Swyft` uses a specific type of simulation-based neural inference techniques, namely, Truncated Marginal Neural Ratio Estimation [@miller2021truncated]. This method improves on standard Markov chain Monte Carlo (MCMC) methods for ABC by learning the likelihood-to-evidence ratio with neural density estimators. Finally, the `Lampe` library provides implementations for a subset of the methods for posterior estimation in the `sbi` library. `Lampe` aims to expose all components (e.g., network architectures, optimizers) in order to provide a flexible and customizable interface for creating neural approximators. All of these libraries are built on top of `PyTorch`.

# Availability, Development, and Documentation

`BayesFlow` is available through PyPI via `pip install bayesflow`. The documentation is hosted on readthedocs. GitHub Actions manage continuous integration through automated code testing. Currently, `BayesFlow` features seven tutorial notebooks. Tutorials 1 and 2 employ toy models to showcase the library's core functionalities, whereas the remaining tutorials are set in applied scenarios. Likewise, tutorials 1-5 revolve around posterior estimation, and tutorials 6 and 7 illustrate model comparison workflows:

1. **Quickstart amortized posterior estimation:** Introduces the basic components of a `BayesFlow` workflow for amortized posterior estimation of a simple multivariate Gaussian model.
2. **Detecting model misspecification in posterior inference:** Demonstrates the integration of misspecification detection into a neural estimator, the diagnosis of misspecification on observed data, and the analysis of the estimator's sensitivity towards model misspecification.
3. **Principled Bayesian workflow for cognitive models:** Shows how `BayesFlow` can be used to derive inferences from intractable models of cognition, in this case evidence accumulation models of decision-making.
4. **Posterior estimation for ODEs:** Demonstrates the application of `BayesFlow` to estimate the parameters of ordinary differential equation (ODE) systems, including the adaptation of the neural network's structure to time-dependent data.
5. **Posterior estimation for SIR-like models:** Extends the analysis of time-series data to epidemiological models of disease outbreak dynamics, including empirical data from the COVID-19 pandemic in Germany.
6. **Bayesian Model comparison:** Introduces the `BayesFlow` workflow for comparing competing stochastic models, illustrated with models of human memory.
7. **Hierarchical Bayesian model comparison:** Extends the previous model comparison scenario to hierarchical models, which allow for considering the evidence contained in exchangeable nested data simultaneously.


# Acknowledgments

We acknowledge contributions from Ulf Mertens and Marco D'Alessandro. The work was partially funded by the Cyber Valley Research Fund (grant number: CyVy-RF-2021-16) and the Deutsche Forschungsgemeinschaft (DFG, German Research Foundation) under Germany’s Excellence Strategy -– EXC-2181 - 390900948 (the Heidelberg Cluster of Excellence STRUCTURES) and EXC-2075 - 390740016 (the Stuttgart Cluster of Excellence SimTech).

# References
