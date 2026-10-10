import copy

import numpy as np

from bayesflow.experimental.graphical_approximator_new.batches import bucket, pad


def test_bucket():
    sizes = set()

    for n in range(1, 1025):
        assert bucket(n) > n
        sizes.add(bucket(n))

    assert len(sizes) <= 4 * 11


def test_pad(two_level_simulator, two_level_approximator):
    batch = two_level_approximator.to_batch(two_level_simulator.sample(3))
    padded = pad(copy.deepcopy(batch))

    for node, entries in batch.items():
        rows = len(entries["indices"][node])
        size = bucket(rows)

        for column in entries["indices"]:
            assert len(padded[node]["indices"][column]) == size
            np.testing.assert_array_equal(padded[node]["indices"][column][:rows], entries["indices"][column])

        for variable, values in entries["values"].items():
            assert len(padded[node]["values"][variable]) == size
            np.testing.assert_array_equal(padded[node]["values"][variable][:rows], values)

        weights = padded[node]["weights"]
        assert np.all(weights[rows:] == 0)
        np.testing.assert_allclose(np.mean(weights), 1.0, rtol=1e-6)
