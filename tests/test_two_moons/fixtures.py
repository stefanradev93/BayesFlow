
import keras
import numpy as np
import pytest

import bayesflow.experimental as bf


@pytest.fixture(scope="module")
def two_moons_global_context():
    return None


@pytest.fixture(scope="module")
def two_moons_local_context():
    @bf.distribution
    def local_context():
        return dict(
            r=np.random.normal(0.1, 0.01),
            alpha=np.random.uniform(-0.5, 0.5)
        )

    return local_context


@pytest.fixture(scope="module")
def two_moons_prior():
    @bf.distribution
    def prior():
        return dict(theta=np.random.normal(-1.0, 1.0, size=2))

    return prior


@pytest.fixture(scope="module")
def two_moons_likelihood():
    @bf.distribution
    def likelihood(r, alpha, theta):
        x1 = -np.abs(theta[0] + theta[1]) / np.sqrt(2.0) + r * np.cos(alpha) + 0.25
        x2 = (-theta[0] + theta[1]) / np.sqrt(2.0) + r * np.sin(alpha)
        return dict(x=np.stack([x1, x2], axis=1))

    return likelihood


@pytest.fixture(scope="module")
def two_moons_generative_model(two_moons_global_context, two_moons_local_context, two_moons_prior, two_moons_likelihood):
    return bf.GenerativeModel(
        global_context=two_moons_global_context,
        local_context=two_moons_local_context,
        prior=two_moons_prior,
        likelihood=two_moons_likelihood,
    )


@pytest.fixture(scope="module")
def two_moons_online_dataset(two_moons_generative_model):
    return bf.datasets.OnlineDataset(two_moons_generative_model, workers=4, use_multiprocessing=True, max_queue_size=16)


@pytest.fixture(scope="module")
def two_moons_inference_network():
    class Subnet(keras.Layer):
        def __init__(self, in_features, out_features):
            super().__init__()
            self.network = keras.Sequential([
                keras.layers.Input(shape=(in_features,)),
                keras.layers.Dense(32, activation="relu"),
                keras.layers.Dense(out_features, kernel_initializer=keras.initializers.Zeros(), bias_initializer=keras.initializers.Zeros())
            ])

        def call(self, x):
            return self.network(x)

    return bf.networks.CouplingFlow.uniform(
        subnet_constructor=Subnet,
        features=2,
        conditions=0,
        layers=2,
        transform="affine",
        base_distribution="normal",
    )


@pytest.fixture(scope="module")
def two_moons_amortized_posterior(two_moons_inference_network):
    return bf.AmortizedPosterior(
        inference_network=two_moons_inference_network
    )
