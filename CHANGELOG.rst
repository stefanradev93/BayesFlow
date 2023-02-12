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

