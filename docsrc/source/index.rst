BayesFlow
=========

Welcome to our BayesFlow library for efficient simulation-based Bayesian workflows!
Our library enables users to create specialized neural networks for amortized Bayesian inference,
which repay users with rapid statistical inference after a potentially longer simulation-based training phase.

.. image:: _static/bayesflow_software_overview.png
    :width: 100%
    :align: center
    :alt: BayesFlow defines a formal workflow for data generation, neural approximation, and model criticism.

BayesFlow features four key capabilities to enhance Bayesian workflows:

1. **Amortized posterior estimation:** Train a generative network to efficiently infer full posteriors (i.e., solve the inverse problem) for all existing and future data compatible with a simulation model.
2. **Amortized likelihood estimation:** Train a generative network to efficiently emulate a simulation model (i.e., solve the forward problem) for all possible parameter configurations or interact with external probabilistic programs.
3. **Amortized model comparison:** Train a neural classifier to recognize the "best" model in a set of competing candidates or combine amortized posterior and likelihood estimation to compute Bayesian evidence and out-of-sample predictive performance.
4. **Model misspecification detection:** Ensure that the resulting posteriors are faithful approximations of the otherwise intractable target posterior, even when simulations do not perfectly represent reality.

Installation
------------

.. tab-set::

    .. tab-item:: Users (stable)

       .. code-block:: bash

          pip install bayesflow


    .. tab-item:: Developers (nightly)

       .. code-block:: bash

          pip install git+https://github.com/stefanradev93/bayesflow@Development


BayesFlow requires Python version 3.9 or later.
The installer should automatically choose the appropriate TensorFlow version depending on your operating system.
However, if the installation fails, Tensorflow and Tensorflow-Probability are likely to be the culprit,
and you might consider starting your bug hunt there.
You can find detailed installation instructions for developers :doc:`here <installation>`.


Citation
--------

You can cite BayesFlow along the lines of:

   - We estimated the approximate posterior distribution with neural posterior estimation (NPE; Papamakarios & Murray, 2016) via the BayesFlow software for amortized Bayesian workflows (Radev et al., 2023b).
   - We trained an neural likelihood estimator (NLE; Papamakarios et al., 2019) via the BayesFlow software for amortized Bayesian workflows (Radev et al., 2023b).
   - We sampled from the approximate joint distribution :math:`p(x, \theta)` using jointly amortized neural approximation (JANA; Radev et al., 2023a), as implemented in the BayesFlow software for amortized Bayesian workflows (Radev et al., 2023b).

1. Radev, S. T., Schmitt, M., Schumacher, L., Elsemüller, L., Pratz, V., Schälte, Y., Köthe, U., & Bürkner, P.-C. (2023). BayesFlow: Amortized Bayesian Workflows With Neural Networks. *arXiv:2306.16015*. (`arXiv paper <https://arxiv.org/abs/2306.16015>`__)
2. Radev, S. T., Schmitt, M., Pratz, V., Picchini, U., Köthe, U., & Bürkner, P.-C. (2023). JANA: Jointly Amortized Neural Approximation of Complex Bayesian Models. *39th conference on Uncertainty in Artificial Intelligence*. (`UAI Proceedings <https://openreview.net/forum?id=dS3wVICQrU0>`__)


::

   @misc{radev2023bayesflow,
     title = {BayesFlow: Amortized Bayesian Workflows With Neural Networks},
     author = {Stefan T Radev and Marvin Schmitt and Lukas Schumacher and Lasse Elsem\"{u}ller and Valentin Pratz and Yannik Sch\"{a}lte and Ullrich K\"{o}the and Paul-Christian B\"{u}rkner},
     year = {2023},
     publisher= {arXiv},
     url={https://arxiv.org/abs/2306.16015}
   }

   @inproceedings{radev2023jana,
     title={{JANA}: Jointly Amortized Neural Approximation of Complex Bayesian Models},
     author={Stefan T. Radev and Marvin Schmitt and Valentin Pratz and Umberto Picchini and Ullrich Koethe and Paul-Christian Buerkner},
     booktitle={The 39th Conference on Uncertainty in Artificial Intelligence},
     year={2023},
     url={https://openreview.net/forum?id=dS3wVICQrU0}
   }

Acknowledgments
---------------

We thank the `PyVBMC <https://acerbilab.github.io/pyvbmc/>`__ team for their great open source documentation which heavily inspired our docs.
The BayesFlow development team acknowledges support from:
Deutsche Forschungsgemeinschaft (DFG, German Research Foundation)
under Germany’s Excellence Strategy -– EXC-2181 - 390900948 (the Heidelberg Cluster of Excellence STRUCTURES),
DFG EXC-2075 - 390740016 (the Stuttgart Cluster of Excellence SimTech),
DFG GRK 2277 via the research training group Statistical Modeling in Psychology (SMiP),
the Cyber Valley Research Fund (grant number: CyVy-RF-2021-16),
the Joachim Herz Foundation,
the EMUNE project ("Invertierbare Neuronale Netze für ein verbessertes Verständnis von Infektionskrankheiten", BMBF, 031L0293A-D),
and the Informatics for Life initiative funded by the Klaus Tschira Foundation.

License and Source Code
-----------------------

BayesFlow is released under :mainbranch:`MIT License <LICENSE>`.
The source code is hosted on the public `GitHub repository <https://github.com/stefanradev93/BayesFlow>`__.

Indices
-------

* :ref:`genindex`
* :ref:`modindex`


.. toctree::
   :maxdepth: 0
   :titlesonly:
   :hidden:

   self
   examples
   api/bayesflow
   installation
   contributing
   about