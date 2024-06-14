# Copyright (c) 2022 The BayesFlow Developers

# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:

# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.

# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.


class SimulationError(Exception):
    """Class for an error in simulation."""

    pass


class SummaryStatsError(Exception):
    """Class for error in summary statistics."""

    pass


class LossError(Exception):
    """Class for error in applying loss."""

    pass


class ShapeError(Exception):
    """Class for error in expected shapes."""

    pass


class ConfigurationError(Exception):
    """Class for error in model configuration, e.g. in meta dict"""

    pass


class InferenceError(Exception):
    """Class for error in forward/inverse pass of a neural components."""

    pass


class OperationNotSupportedError(Exception):
    """Class for error that occurs when an operation is demanded but not supported,
    e.g., when a trainer is initialized without generative model but the user demands it to simulate data.
    """

    pass


class ArgumentError(Exception):
    """Class for error that occurs as a result of a function call which is invalid due to the input arguments."""
    pass
