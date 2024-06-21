import keras
import numpy as np


def test_sample_is_random(simulator, batch_size):
    samples = simulator.sample((batch_size,))

    for tensor in samples.values():
        array = keras.ops.convert_to_numpy(tensor)

        expected = keras.ops.size(tensor)
        actual = len(np.unique(array))
        assert actual == expected
