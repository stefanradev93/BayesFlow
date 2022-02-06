class SimulationError(Exception):
    """Class for an error in simulation."""
    pass


class SummaryStatsError(Exception):
    """Class for error in summary statistics."""
    pass


class LossError(Exception):
    """ Class for error in applying loss. """
    pass


class ConfigurationError(Exception):
    """Class for error in model configuration, e.g. in meta dict"""
    pass


class OperationNotSupportedError(Exception):
    """
    Class for error that occurs when an operation is demanded but not supported,
    e.g. when a trainer is initialized without generative model but the user demands it to simulate data.
    """