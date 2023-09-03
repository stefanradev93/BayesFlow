---
title: "BayesFlow: Amortized Bayesian Workflows With Neural Networks"
tags:
  - simulation-based inference
  - likelihood-free inference
  - Bayesian inference
  - amortized Bayesian inference
  - Python
date: "22 June 2023"
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
    orcid: "0000-0003-1293-820X"
  - name: Lukas Schumacher
    affiliation: 3
    orcid: "0000-0003-1512-8288"
  - name: Lasse Elsemüller
    affiliation: 3
    orcid: "0000-0003-0368-720X"
  - name: Valentin Pratz
    affiliation: 4
    orcid: "0000-0001-8371-3417"
  - name: Yannik Schälte
    affiliation: 5
    orcid: "0000-0003-1293-820X"
  - name: Ullrich Köthe
    affiliation: 4
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
  - name: Institute for Psychology, Heidelberg University, Germany
    index: 3
  - name: Visual Learning Lab, Heidelberg University, Germany
    index: 4
  - name: Life and Medical Sciences Institute, University of Bonn, Germany
    index: 5
  - name: Department of Statistics, TU Dortmund University, Germany
    index: 6
---

# Summary
Modern Bayesian inference involves a mixture of computational techniques for estimating, validating, and drawing conclusions from probabilistic models as part of principled workflows for data analysis [@burkner_models_2022; @gelman_bayesian_2020; @schad2021toward]. Typical problems in Bayesian workflows are the approximation of intractable posterior distributions for diverse model types and the comparison of competing models of the same process in terms of their complexity and predictive performance. However, despite their theoretical appeal and utility, the practical execution of Bayesian workflows is often limited by computational bottlenecks: Obtaining even a single posterior may already take a long time, such that repeated estimation for the purpose of model validation or calibration becomes completely infeasible.

`BayesFlow` provides a framework for *simulation-based* training of established neural network architectures, such as transformers [@vaswani2017attention] and normalizing flows [@papamakarios2021normalizing], for *amortized* data compression and inference. *Amortized Bayesian inference* (ABI), as implemented in `BayesFlow`, enables users to train custom neural networks on model simulations and re-use these networks for any subsequent application of the models. Since the trained networks can perform inference almost instantaneously (typically well below one second), the upfront neural network training is quickly amortized. For instance, amortized inference allows us to test a model's ability to recover its parameters [@schad2021toward] or assess its simulation-based calibration [@talts2018; @sailynoja2022graphical] for different data set sizes in a matter of seconds, even though this may require the estimation of thousands of posterior distributions. `BayesFlow` offers a user-friendly API, which encapsulates the details of neural network architectures and training procedures that are less relevant for the practitioner and provides robust default implementations that work well across many applications. At the same time, `BayesFlow` implements a modular software architecture, allowing machine learning scientists to modify every component of the pipeline for custom applications as well as research at the frontier of Bayesian inference.

![`BayesFlow` defines a formal workflow for data generation, neural approximation, and model criticism.\label{fig:figure1}](bayesflow_software_figure1.pdf)

# Statement of Need

`BayesFlow` embodies functionality that is specifically designed for building and validating amortized Bayesian workflows with the help of neural networks. \autoref{fig:figure1} outlines a typical workflow in the context of amortized posterior and likelihood estimation. A simulator coupled with a prior defines a generative Bayesian model. The generative model may depend on various (optional) context variates like varying numbers of observations, design matrices, or positional encodings. The generative scope of the model and the range of context variables determine the *scope of amortization*, that is, over which types of data the neural approximator can be applied without re-training. The neural approximators interact with model outputs (parameters, data) and context variates through a configurator. The configurator is responsible for carrying out transformations (e.g., input normalization, double-to-float conversion, etc.) that are not part of the model but may facilitate neural network training and convergence.

\autoref{fig:figure1} also illustrates an example configuration of four neural networks: 1) a summary network to compress simulation outcomes (individual data points, sets, or time series) into informative embeddings; 2) a posterior network to learn an amortized approximate posterior; and 3) another summary network to compress simulation inputs (parameters) into informative embeddings; and 4) a likelihood network to learn an amortized approximate likelihood. \autoref{fig:figure1} depicts the standalone and joint capabilities of the networks when applied in isolation or in tandem. The input conditions for the posterior and likelihood networks are partitioned by the configurator: Complex ("summary") conditions are processed by the respective summary network into embeddings, while very simple ("direct") conditions can bypass the summary network and flow straight into the neural approximator.

Currently, the software features four key capabilities for enhancing Bayesian workflows, which have been described in the referenced works:

1. **Amortized posterior estimation:** Train a generative network to efficiently infer full posteriors (i.e., solve the inverse problem) for all existing and future data compatible with a simulation model [@radev2020bayesflow]. 
2. **Amortized likelihood estimation:** Train a generative network to efficiently emulate a simulation model (i.e., solve the forward problem) for all possible parameter configurations or interact with external probabilistic programs [@radev2023jana; @boelts2022flexible].
3. **Amortized model comparison:** Train a neural classifier to recognize the "best" model in a set of competing candidates [@radev2020evidential; @schmitt2022meta; @elsemuller2023deep] or combine amortized posterior and likelihood estimation to compute Bayesian evidence and out-of-sample predictive performance [@radev2023jana].
4. **Model misspecification detection:** Ensure that the resulting posteriors are faithful approximations of the otherwise intractable target posterior, even when simulations do not perfectly represent reality [@schmitt2021detecting; @radev2023jana].


`BayesFlow` has been used for amortized Bayesian inference in various areas of applied research, such as epidemiology [@radev2021outbreakflow], cognitive modeling [@von2022mental; @wieschen2020jumping; @sokratous2023ask], computational psychiatry [@d2020bayesian], neuroscience [@ghaderi2022general], particle physics [@bieringer2021measuring], agent-based econometrics models [@shiono2021estimation], seismic imaging [@siahkoohi2023reliable], user behavior [@moon2023amortized], structural health monitoring [@zeng2023probabilistic], aerospace [@tsilifis2022inverse] and wind turbine design [@noever2022model], micro-electro-mechanical systems testing [@heringhaus2022towards], and fractional Brownian motion [@verdier2022variational].

The software is built on top of `TensorFlow` [@abadi2016tensorflow] and thereby enables off-the-shelf support for GPU and TPU acceleration. Furthermore, it can seamlessly interact with TensorFlow Probability [@dillon2017tensorflow] for flexible latent distributions and a variety of joint priors.

# Related Software
When a non-amortized inference procedure does not create a computational bottleneck, approximate Bayesian computation (ABC) might be an appropriate tool. This is the case if a single data set needs to be analyzed, if an infrastructure for parallel computing is readily available, or if repeated re-fits of a model (e.g., cross-validation) are not desired.
A variety of mature Python packages for ABC exist, such as PyMC [@Salvatier2016], pyABC [@schaelte2022pyabc], ABCpy [@dutta2021abcpy], or ELFI [@lintusaari2018elfi]. In contrast to these packages, `BayesFlow` focuses on amortized inference, but can also interact with ABC samplers (e.g., use BayesFlow to learn informative summary statistics for an ABC analysis).

When it comes to simulation-based inference with neural networks, the `sbi` toolkit enables both likelihood and posterior estimation using different inference algorithms, such as Neural Posterior Estimation [@papamakarios2021normalizing], Sequential Neural Posterior Estimation [@greenberg2019automatic] and Sequential Neural Likelihood Estimation [@papamakarios2019sequential]. `BayesFlow` and `sbi` can be viewed as complementary toolkits, where `sbi` implements a variety of different approximators for standard modeling scenarios, while `BayesFlow` focuses on amortized workflows with user-friendly default settings and optional customization. The `Swyft` library focuses on Bayesian parameter inference in physics and astronomy. `Swyft` uses a specific type of simulation-based neural inference technique, namely, Truncated Marginal Neural Ratio Estimation [@miller2021truncated]. This method improves on standard Markov chain Monte Carlo (MCMC) methods for ABC by learning the likelihood-to-evidence ratio with neural density estimators. Finally, the `Lampe` library provides implementations for a subset of the methods for posterior estimation in the `sbi` library, aiming to expose all components (e.g., network architectures, optimizers) in order to provide a customizable interface for creating neural approximators. All of these libraries are built on top of `PyTorch`.

# Availability, Development, and Documentation

`BayesFlow` is available through PyPI via `pip install bayesflow`, the development version is available via GitHub. GitHub Actions manage continuous integration through automated code testing and documentation. The documentation is hosted at [www.bayesflow.org](https://bayesflow.org/). Currently, `BayesFlow` features seven tutorial notebooks. These notebooks showcase different aspects of the software, ranging from toy examples to applied modeling scenarios, and illustrating both posterior estimation and model comparison workflows.


# Acknowledgments

We thank Ulf Mertens, Marco D'Alessandro, René Bucchia, The-Gia Leo Nguyen, Jonas Arruda, Lea Zimmermann, and Leonhard Volz for contributing to the GitHub repository. STR was funded by the Deutsche Forschungsgemeinschaft (DFG, German Research Foundation) under Germany’s Excellence Strategy - EXC-2181 - 390900948 (the Heidelberg Cluster of Excellence STRUCTURES), MS and PCB were supported by the Cyber Valley Research Fund (grant number: CyVy-RF-2021-16) and the DFG EXC-2075 - 390740016 (the Stuttgart Cluster of Excellence SimTech). LS and LE were supported by a grant from the DFG (GRK 2277) to the research training group Statistical Modeling in Psychology (SMiP). YS acknowledges support from the Joachim Herz Foundation. UK was supported by the Informatics for Life initiative funded by the Klaus Tschira Foundation. YS and UK were supported by the EMUNE project ("Invertierbare Neuronale Netze für ein verbessertes Verständnis von Infektionskrankheiten", BMBF, 031L0293A-D).

# References
