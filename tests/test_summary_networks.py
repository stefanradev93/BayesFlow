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

import numpy as np
import tensorflow as tf

import pytest

from bayesflow.summary_networks import *


def test_equivariant_module():
    """This function tests the permutation equivariance property of the `EquivariantModule`."""

    pass


def test_invariant_module():
    """This function tests the permutation invariance property of the `InvariantModule`."""
    pass


def test_invariant_network():
    """This function tests the fidelity of the invariant network with a couple of relevant
    configurations w.r.t. permutation invariance and output dimensions."""

    pass


def test_multi_conv1d():
    """This function tests the fidelity of the `MultiConv1D` module w.r.t. output dimensions
    using a number of relevant configurations."""

    pass


def test_multi_conv_network():
    """This function tests the fidelity of the `MultiConvNetwork` w.r.t. output dimensions
    using a number of relevant configurations."""

    pass