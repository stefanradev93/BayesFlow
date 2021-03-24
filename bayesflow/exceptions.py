class SimulationError(Exception):
    """Class for an error in simulation."""
    pass


class SummaryStatsError(Exception):
    """Class for error in summary statistics."""
    pass


class ConfigurationError(Exception):
    """Class for error in model configuration, e.g. in meta dict"""
    pass
