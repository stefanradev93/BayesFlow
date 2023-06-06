Change log
==========


1.0.2b Series
----------

Minor

1. Added option for residual networks in coupling blocks using a ``residual=True`` in the coupling layer config.

1.0.3b Series
----------

Major (Breaking)

1. Coupling layers were refactored to ensure consistency. Old checkpoints may no longer load.

Minor (Features)

1. Added ``attention.py`` module containing helper networks for building transformers
2. Added ``SetTransformer`` class in ``summary_networks.py`` as a viable alternative to ``DeepSet`` summary networks.
3. Added ``TimeSeriesTransformer`` class in ``summary_networks.py`` as a viable alternative to ``SequentialNetworks`` summary networks.
4. Added ``plot_z_score_contraction()`` diagnostic in ``diagnostics.py`` for gauging global inferential adequacy
5. Added ``Orthogonal`` in ``helper_networks.py`` for learnable generalized permutations.

1.1 Series
----------

Major (Breaking)

1. Coupling layers have been refactored to ensure easy interoperability between spline flows and affine coupling flows
2. New internal classes and layers have been added! Saving and loading of old models will not work! However, the interface
remains consistent.
3. Model comparison now works for both hierarchical and non-hierarchical Bayesian models. Classes have been generalized
and semantics go beyond the ``EvidentialNetwork``
4. Default settings have been changed to reflect recent insights into better hyperparameter settings.

Minor

Features:
1. Added option for ``permutation='learnable'`` when creating an ``InvertibleNetwork``
2. Added option for ``coupling_design in ["affine", "spline", "interleaved"]`` when creating an ``InvertibleNetwork``
3. Simplified passing additional settings to the internal networks. For instance, you
can now simply do
``inference_network = InvertibleNetwork(num_params=20, coupling_net_settings={'mc_dropout': True})``
to get a Bayesian neural network.
4. ``PMPNetwork`` has been added for model comparison according to findings in https://arxiv.org/abs/2301.11873
5. ``HierarchicalNetwork`` wrapper has been added to act as a summary network for hierarchical Bayesian models according to
https://arxiv.org/abs/2301.11873
6. Publication-ready calibration diagnostic for expected calibration error (ECE) in a model comparison setting has been
added to ``diagnostics.py`` and is accessible as ``plot_calibration_curves()``
7. A new module ``experimental`` has been added currently containing ``rectifiers.py``.
8. Default settings for transformer-based architectures.
9. Numerical calibration error using ``posterior_calibration_error()``

General Improvements:
1. Improved docstrings and consistent use of keyword arguments vs. configuration dictionaries
2. Increased focus on transformer-based architectures as summary networks
3. Figures resulting ``diagnostics.py`` have been improved and prettified
4. Added a module ``sensitivity.py`` for testing the sensitivity of neural approximators to model misspecification
5. Multiple bugfixes, including a major bug affecting the saving and loading of learnable permutations
