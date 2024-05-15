import math

import keras
import pytest
from keras import ops as K
from keras import random as R

import bayesflow.experimental as bf


@pytest.fixture()
def prior():
    class Prior:
        def sample(self, batch_shape):
            r = R.normal(shape=batch_shape + (1,), mean=0.1, stddev=0.01)
            alpha = R.uniform(shape=batch_shape + (1,), minval=-0.5 * math.pi, maxval=0.5 * math.pi)
            theta = R.uniform(shape=batch_shape + (2,), minval=-1.0, maxval=1.0)

            return dict(r=r, alpha=alpha, theta=theta)

    return Prior()


@pytest.fixture()
def likelihood():
    class Likelihood:
        def sample(self, batch_shape, r, alpha, theta):
            x1 = -K.abs(theta[0] + theta[1]) / K.sqrt(2.0) + r * K.cos(alpha) + 0.25
            x2 = (-theta[0] + theta[1]) / K.sqrt(2.0) + r * K.sin(alpha)
            return dict(x=K.stack([x1, x2], axis=-1))

    return Likelihood()


@pytest.fixture()
def joint_distribution(prior, likelihood):
    return bf.simulation.JointDistribution(prior, likelihood)


@pytest.fixture()
def dataset(joint_distribution):
    # TODO: do not use hard-coded batch size
    return bf.datasets.OnlineDataset(joint_distribution, workers=4, use_multiprocessing=True, max_queue_size=16, batch_size=16)


@pytest.fixture()
def inference_network():
    return bf.networks.CouplingFlow.all_in_one(
        subnet_builder="default",
        target_dim=2,
        num_layers=2,
        transform="affine",
        base_distribution="normal",
    )


@pytest.fixture()
def summary_network():
    return None


@pytest.fixture()
def amortizer(inference_network, summary_network):
    return bf.AmortizedPosterior(inference_network, summary_network)
