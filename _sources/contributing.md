Contributing to BayesFlow
=========================

Workflow
--------

If you start working on a new feature or a bug fix, please create an issue on GitHub 
describing the issue and assign yourself.

Make sure that you are working in the ``Development`` branch, which contains the latest updates.
Create an own branch or use a fork, where you can implement your changes. To merge your work, please follow these steps:

1. Create a pull request to the ``Development`` branch with a short summary of the code changes;
2. Check that code changes are covered by tests, and all tests pass;
3. Check that the documentation is up-to-date;
4. Request a code review from the main developers.

Environment
-----------

If you want to contribute to the development of BayesFlow, you need to install developer requirements via:

    pip install -r requirements_dev.txt

Tox
---

Second, this installs the virtual testing tool `tox`
<https://tox.readthedocs.io/en/latest/>, which we use for all tests, formatting and quality checks. Its configuration is specified in ``tox.ini``.
To run it locally, simply execute:

    tox [-e flake8]

with an optional ``-e`` flag specifying the environments to run. See ``tox.ini`` for more details.

Pre-commit hooks
----------------

Pre-commit hooks are used to examine the fidelity of the code that's about to be committed.
For this, you need to install a pre-commit tool (e.g., <https://pre-commit.com/>).

To add those hooks to the ``.git`` folder of your local clone such that they are automatically executed on every commit, run::

    pre-commit install

When adding new hooks, consider manually running ``pre-commit run --all-files`` once, as usually only the diff is checked. The configuration is specified in
``.pre-commit-config.yaml``.

GitHub Actions
--------------

We use GitHub Actions for automatic continuous integration testing (<https://github.com/features/actions?>). All tests are run on pull requests and required to pass. 
The configuration is specified in ``.github/workflows/tests.yml``.

Unit tests
----------

Unit tests are located in the ``tests`` folder. All files starting with ``test_`` contain tests and are automatically run via GitHub Actions.
We use `pytest` (<https://docs.pytest.org/en/latest/>) for our testing environment.

You can run the all tests locally via:

    pytest -e

Or a specific test via:

    pytest -e test_[mytest]

Tutorial Notebooks
------------------

New tutorial notebooks are always welcome! You can add your tutorial notebook file to `examples/` and add a reference 
to the list of notebooks in `docsrc/source/examples.rst`. 
Re-build the documentation (see below) and your notebook will be included.

Documentation
-------------

The documentation uses [sphinx](https://www.sphinx-doc.org/) and relies on [numpy style docstrings](https://numpydoc.readthedocs.io/en/latest/format.html) in classes and functions.
The overall *structure* of the documentation is manually designed. This also applies to the API documentation. This has two implications for you:

1. If you add to existing submodules, the documentation will update automatically (given that you use proper numpy docstrings).
2. If you add a new submodule or subpackage, you need to add a file to `docsrc/source/api` and a reference to the new module to the appropriate section of `docsrc/source/api/bayesflow.rst`.

You can re-build the documentation with

    cd docsrc/
    make clean && make github

The entry point of the rendered documentation will be at `docs/index.html`.