import keras
import numpy as np


def test_sample_is_random(simulator, batch_size):
    samples = simulator.sample((batch_size,))

    for tensor in samples.values():
        array = keras.ops.convert_to_numpy(tensor)

        expected = keras.ops.size(tensor)
        actual = np.size(np.unique(array))
        assert actual == expected


def test_sequential_simulators(sequential_two_moons, batch_size):
    data = sequential_two_moons.sample((batch_size,))

    # Test all keys are present
    result_keys = set([key for key in data.keys()])
    expected_keys = set(["r", "alpha", "theta", "x"])
    assert result_keys == expected_keys

    # Test correct output shapes are returned
    assert data["r"].shape == (batch_size,)
    assert data["alpha"].shape == (batch_size,)
    assert data["theta"].shape == (batch_size, 2)
    assert data["x"].shape == (batch_size, 2)
