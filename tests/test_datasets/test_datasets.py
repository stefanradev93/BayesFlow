
import keras
import pickle
import pytest


def test_dataset_is_picklable(dataset):
    pickled = pickle.loads(pickle.dumps(dataset))

    assert type(pickled) is type(dataset)

    samples = dataset[0]  # dict of {param_name: param_value}
    samples = next(iter(samples.values()))  # first param value

    pickled_samples = pickled[0]
    pickled_samples = next(iter(pickled_samples.values()))

    assert keras.ops.shape(samples) == keras.ops.shape(pickled_samples)


def test_dataset_works_in_fit(model, dataset):
    model.fit(dataset, epochs=1, steps_per_epoch=1)


def test_dataset_returns_batch(dataset, batch_size):
    samples = dataset[0]  # dict of {param_name: param_value}
    samples = next(iter(samples.values()))  # first param value

    assert keras.ops.shape(samples)[0] == batch_size
