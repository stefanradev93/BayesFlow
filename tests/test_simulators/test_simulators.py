import keras


def test_sample(simulator, batch_size):
    samples = simulator.sample((batch_size,))

    # test output structure
    assert isinstance(samples, dict)

    for key, value in samples.items():
        print(f"{key}.shape = {keras.ops.shape(value)}")

        # test type
        assert keras.ops.is_tensor(value)
        assert keras.utils.standardize_dtype(value.dtype) == "float32"

        # test shape
        assert keras.ops.shape(value)[0] == batch_size
        assert keras.ops.ndim(value) > 1

        # test batch randomness
        assert keras.ops.any(~keras.ops.isclose(value, value[0]))
