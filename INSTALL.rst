Install
=======

Requirements
------------

This package requires Python 3.9 or later.
A simple installation is possible via `Miniconda <https://docs.conda.io/en/latest/miniconda.html>`_,
e.g. (on Linux)

.. code-block:: bash

   CONDA_DIR=$PWD/miniconda3
   wget -O Miniconda3.sh https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh
   bash Miniconda3.sh -b -p $CONDA_DIR
   rm Miniconda3.sh

Thereafter, activate the conda environment via

.. code-block:: bash

   eval "$($CONDA_DIR/bin/conda shell.bash hook)"

(Run ``conda init`` once to automatically load conda in interactive shells.)
Having installed and activated conda, optionally you can create a local environment via

.. code-block:: bash

   conda create -y -n bf python=3.10

and activate via

.. code-block:: bash

   conda activate bf

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
