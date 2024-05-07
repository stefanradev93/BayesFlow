
from .fixtures import *


def test_fit(two_moons_amortized_posterior, two_moons_online_dataset):
    # TODO: verify the model learns something by comparing a metric before and after training
    two_moons_amortized_posterior.fit(two_moons_online_dataset, epochs=10, steps_per_epoch=10, batch_size=32)
