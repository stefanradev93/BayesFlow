# BayesFlow <img src="img/bayesflow_hex.png" style="float: right; width: 20%; height: 20%;" align="right" alt="BayesFlow Logo" />
![GitHub Actions Workflow Status](https://img.shields.io/github/actions/workflow/status/stefanradev93/bayesflow/tests.yaml?style=for-the-badge&label=Tests)
![Codecov](https://img.shields.io/codecov/c/github/stefanradev93/bayesflow/streamlined-backend?style=for-the-badge)
[![DOI](https://img.shields.io/badge/DOI-10.21105%2Fjoss.05702-blue?style=for-the-badge)](https://doi.org/10.21105/joss.05702)
![PyPI - License](https://img.shields.io/pypi/l/bayesflow?style=for-the-badge)

BayesFlow is a Python library for simulation-based **Amortized Bayesian Inference** with neural networks.
It provides users with:

- A user-friendly API for rapid Bayesian workflows
- A rich collection of neural network architectures
- Multi-Backend Support: [PyTorch](https://github.com/pytorch/pytorch), [TensorFlow](https://github.com/tensorflow/tensorflow), [JAX](https://github.com/google/jax), and [NumPy](https://github.com/numpy/numpy)

BayesFlow is designed to be a flexible and efficient tool, enabling rapid statistical inference after a
potentially longer simulation-based training phase.


## Install

### Backend

First, install your machine learning backend of choice. Note that BayesFlow **will not run** without a backend.

Once installed, set the appropriate backend environment variable. For example, to use PyTorch:

```bash
export KERAS_BACKEND=torch
```

If you use conda, you can instead set this individually for each environment:

```bash
conda env config vars set KERAS_BACKEND=torch
```

### Using Conda

We recommend installing with conda (or mamba).

```bash
conda install -c conda-forge bayesflow
```

### Using pip

```bash
pip install bayesflow
```

### From Source

Stable version:

```bash
git clone https://github.com/stefanradev93/bayesflow
cd bayesflow
conda env create --file environment.yaml --name bayesflow
```

Development version:

```bash
git clone https://github.com/stefanradev93/bayesflow
cd bayesflow
git checkout dev
conda env create --file environment.yaml --name bayesflow
```


## Getting Started

Check out some of our walk-through notebooks:

1. [Quickstart amortized posterior estimation](examples/Intro_Amortized_Posterior_Estimation.ipynb)
2. [Tackling strange bimodal distributions](examples/TwoMoons_Bimodal_Posterior.ipynb)
3. [Detecting model misspecification in posterior inference](examples/Model_Misspecification.ipynb)
4. [Principled Bayesian workflow for cognitive models](examples/LCA_Model_Posterior_Estimation.ipynb)
5. [Posterior estimation for ODEs](examples/Linear_ODE_system.ipynb)
6. [Posterior estimation for SIR-like models](examples/Covid19_Initial_Posterior_Estimation.ipynb)
7. [Model comparison for cognitive models](examples/Model_Comparison_MPT.ipynb)
8. [Hierarchical model comparison for cognitive models](examples/Hierarchical_Model_Comparison_MPT.ipynb)
9. [Posterior estimation using an SBML-model](examples/SBML_Model_Posterior_Estimation.ipynb)


## Documentation \& Help

Documentation is available at https://bayesflow.org. Please use the [BayesFlow Forums](https://discuss.bayesflow.org/) for any BayesFlow-related questions and discussions, and [GitHub Issues](https://github.com/stefanradev93/BayesFlow/issues) for bug reports and feature requests.


## Conceptual Overview

A cornerstone idea of amortized Bayesian inference is to employ generative
neural networks for parameter estimation, model comparison, and model validation
when working with intractable simulators whose behavior as a whole is too
complex to be described analytically. The figure below presents a higher-level
overview of neurally bootstrapped Bayesian inference.

<img src="https://github.com/stefanradev93/BayesFlow/blob/master/img/high_level_framework.png?raw=true" width=80% height=80%>


### References and Further Reading

- Radev S. T., D’Alessandro M., Mertens U. K., Voss A., Köthe U., & Bürkner P.
C. (2021). Amortized Bayesian Model Comparison with Evidental Deep Learning.
<em>IEEE Transactions on Neural Networks and Learning Systems</em>.
doi:10.1109/TNNLS.2021.3124052 available for free at: https://arxiv.org/abs/2004.10629

- Schmitt, M., Radev, S. T., & Bürkner, P. C. (2022). Meta-Uncertainty in
Bayesian Model Comparison. In <em>International Conference on Artificial Intelligence
and Statistics</em>, 11-29, PMLR, available for free at: https://arxiv.org/abs/2210.07278

- Elsemüller, L., Schnuerch, M., Bürkner, P. C., & Radev, S. T. (2023). A Deep
Learning Method for Comparing Bayesian Hierarchical Models. <em>ArXiv preprint</em>,
available for free at: https://arxiv.org/abs/2301.11873

- Radev, S. T., Schmitt, M., Pratz, V., Picchini, U., Köthe, U., & Bürkner, P.-C. (2023).
JANA: Jointly amortized neural approximation of complex Bayesian models.
*Proceedings of the Thirty-Ninth Conference on Uncertainty in Artificial Intelligence, 216*, 1695-1706.
([arXiv](https://arxiv.org/abs/2302.09125))([PMLR](https://proceedings.mlr.press/v216/radev23a.html))

## Support

This project is currently managed by researchers from Rensselaer Polytechnic Institute, TU Dortmund University, and Heidelberg University. It is partially funded by the Deutsche Forschungsgemeinschaft (DFG, German Research Foundation, Project 528702768). The project is further supported by Germany's Excellence Strategy -- EXC-2075 - 390740016 (Stuttgart Cluster of Excellence SimTech) and EXC-2181 - 390900948 (Heidelberg Cluster of Excellence STRUCTURES), as well as the Informatics for Life initiative funded by the Klaus Tschira Foundation.

## Citing BayesFlow

You can cite BayesFlow along the lines of:

- We approximated the posterior with neural posterior estimation and learned summary statistics (NPE; Radev et al., 2020), as implemented in the BayesFlow software for amortized Bayesian workflows (Radev et al., 2023a).
- We approximated the likelihood with neural likelihood estimation (NLE; Papamakarios et al., 2019) without hand-crafted summary statistics, as implemented in the BayesFlow software for amortized Bayesian workflows (Radev et al., 2023b).
- We performed simultaneous posterior and likelihood estimation with jointly amortized neural approximation (JANA; Radev et al., 2023a), as implemented in the BayesFlow software for amortized Bayesian workflows (Radev et al., 2023b).

1. Radev, S. T., Schmitt, M., Schumacher, L., Elsemüller, L., Pratz, V., Schälte, Y., Köthe, U., & Bürkner, P.-C. (2023a). BayesFlow: Amortized Bayesian workflows with neural networks. *The Journal of Open Source Software, 8(89)*, 5702.([arXiv](https://arxiv.org/abs/2306.16015))([JOSS](https://joss.theoj.org/papers/10.21105/joss.05702))
2. Radev, S. T., Mertens, U. K., Voss, A., Ardizzone, L., Köthe, U. (2020). BayesFlow: Learning complex stochastic models with invertible neural networks. *IEEE Transactions on Neural Networks and Learning Systems, 33(4)*, 1452-1466. ([arXiv](https://arxiv.org/abs/2003.06281))([IEEE TNNLS](https://ieeexplore.ieee.org/document/9298920))
3. Radev, S. T., Schmitt, M., Pratz, V., Picchini, U., Köthe, U., & Bürkner, P.-C. (2023b). JANA: Jointly amortized neural approximation of complex Bayesian models. *Proceedings of the Thirty-Ninth Conference on Uncertainty in Artificial Intelligence, 216*, 1695-1706. ([arXiv](https://arxiv.org/abs/2302.09125))([PMLR](https://proceedings.mlr.press/v216/radev23a.html))

**BibTeX:**

```
@article{bayesflow_2023_software,
  title = {{BayesFlow}: Amortized {B}ayesian workflows with neural networks},
  author = {Radev, Stefan T. and Schmitt, Marvin and Schumacher, Lukas and Elsemüller, Lasse and Pratz, Valentin and Schälte, Yannik and Köthe, Ullrich and Bürkner, Paul-Christian},
  journal = {Journal of Open Source Software},
  volume = {8},
  number = {89},
  pages = {5702},
  year = {2023}
}

@article{bayesflow_2020_original,
  title = {{BayesFlow}: Learning complex stochastic models with invertible neural networks},
  author = {Radev, Stefan T. and Mertens, Ulf K. and Voss, Andreas and Ardizzone, Lynton and K{\"o}the, Ullrich},
  journal = {IEEE transactions on neural networks and learning systems},
  volume = {33},
  number = {4},
  pages = {1452--1466},
  year = {2020}
}

@inproceedings{bayesflow_2023_jana,
  title = {{JANA}: Jointly amortized neural approximation of complex {B}ayesian models},
  author = {Radev, Stefan T. and Schmitt, Marvin and Pratz, Valentin and Picchini, Umberto and K\"othe, Ullrich and B\"urkner, Paul-Christian},
  booktitle = {Proceedings of the Thirty-Ninth Conference on Uncertainty in Artificial Intelligence},
  pages = {1695--1706},
  year = {2023},
  volume = {216},
  series = {Proceedings of Machine Learning Research},
  publisher = {PMLR}
}
```
