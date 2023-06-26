*********
BayesFlow
*********

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
############

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

How To
######

.. toctree::
   :maxdepth: 1
   :titlesonly:

   examples
   documentation
   installation

License and Source Code
#######################

BayesFlow is released under :mainbranch:`MIT License <LICENSE>`.
The source code is hosted on the public `GitHub repository <https://github.com/stefanradev93/BayesFlow>`_.

Acknowledgments
###############

We thank the `PyVBMC <https://acerbilab.github.io/pyvbmc/>`_ team for their great open source documentation which heavily inspired our docs.
The BayesFlow development team acknowledges support from:
Deutsche Forschungsgemeinschaft (DFG, German Research Foundation)
under Germany’s Excellence Strategy -– EXC-2181 - 390900948 (the Heidelberg Cluster of Excellence STRUCTURES),
DFG EXC-2075 - 390740016 (the Stuttgart Cluster of Excellence SimTech),
DFG GRK 2277 via the research training group Statistical Modeling in Psychology (SMiP),
the Cyber Valley Research Fund (grant number: CyVy-RF-2021-16),
the Joachim Herz Foundation,
the EMUNE project ("Invertierbare Neuronale Netze für ein verbessertes Verständnis von Infektionskrankheiten", BMBF, 031L0293A-D),
and the Informatics for Life initiative funded by the Klaus Tschira Foundation.


.. toctree::
   :maxdepth: 1
   :titlesonly:
   :hidden:

   about