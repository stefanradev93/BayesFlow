import keras
import pickle


def test_dataset_is_picklable(dataset):
    pickled = pickle.loads(pickle.dumps(dataset))

    assert type(pickled) is type(dataset)

    samples = dataset[0]  # dict of {param_name: param_value}
    pickled_samples = pickled[0]
    assert isinstance(samples, dict)
    assert isinstance(pickled_samples, dict)

    assert list(samples.keys()) == list(pickled_samples.keys())

    for key in samples.keys():
        assert keras.ops.shape(samples[key]) == keras.ops.shape(pickled_samples[key])


def test_dataset_works_in_fit(model, dataset):
    print(next(iter(dataset[0].values())).dtype)
    model.fit(dataset, epochs=1, steps_per_epoch=1)


def test_dataset_returns_batch(dataset, batch_size):
    samples = dataset[0]  # dict of {param_name: param_value}
    samples = next(iter(samples.values()))  # first param value

    assert keras.ops.shape(samples)[0] == batch_size
