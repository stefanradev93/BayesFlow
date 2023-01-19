Install
=======

Requirements
------------

This package requires Python 3.9 or later.
A simple installation is possible via `Miniconda <https://docs.conda.io/en/latest/miniconda.html>`_,
e.g. (on Linux)

.. code-block:: bash

   wget https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh
   bash Miniconda3-latest-Linux-x86_64.sh
   rm Miniconda3-latest-Linux-x86_64.sh

Having installed and activated conda, optionally a local environment can be created via

.. code-block:: bash

   conda create -y -n bayesflow python=3.10

Install from GitHub
-------------------

Install BayesFlow from GitHub via

.. code-block:: bash

   pip install git+https://github.com/stefanradev93/bayesflow


If you want to work with the latest development version (which may however be unstable), instead use

.. code-block:: bash

   pip install git+https://github.com/stefanradev93/bayesflow@Development

If you need access to the source code, instead use

.. code-block:: bash

   git clone git@github.com:stefanradev93/bayesflow.git
   cd bayesflow
   pip install -e .
