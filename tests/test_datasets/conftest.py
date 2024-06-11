
import keras
import pytest


@pytest.fixture()
def batch_size():
    return 16


@pytest.fixture(params=["online_dataset", "offline_dataset"])
def dataset(request, online_dataset, offline_dataset):
    return request.getfixturevalue(request.param)


@pytest.fixture()
def model():
    class Model(keras.Model):
        def call(self, *args, **kwargs):
            pass

        def compute_loss(self, **kwargs):
            return keras.ops.zeros(())

    model = Model()
    model.compile(optimizer=None)

    return model


@pytest.fixture()
def offline_dataset(simulator, batch_size, workers, use_multiprocessing):
    from bayesflow.experimental import OfflineDataset

    # TODO: there is a bug in keras where if len(dataset) == 1 batch
    #  fit will error because no logs are generated
    #  the single batch is then skipped entirely
    data = simulator.sample((batch_size * 2,))
    return OfflineDataset(data, batch_size=batch_size, workers=workers, use_multiprocessing=use_multiprocessing)


@pytest.fixture()
def online_dataset(simulator, batch_size, workers, use_multiprocessing):
    from bayesflow.experimental import OnlineDataset

    return OnlineDataset(simulator, batch_size=batch_size, workers=workers, use_multiprocessing=use_multiprocessing)


# needs to be global for pickle to work

from bayesflow.experimental.simulation.decorators.distribution_decorator import DistributionDecorator as make_distribution


class Simulator:
    def sample(self, batch_shape):
        return dict(x=keras.random.normal(batch_shape + (2,)))


@make_distribution(is_batched=True)
def batched_decorated_simulator(batch_shape):
    return dict(x=keras.random.normal(batch_shape + (2,)))


@make_distribution(is_batched=False)
def unbatched_decorated_simulator():
    return dict(x=keras.random.normal((2,)))


@pytest.fixture(params=["class", "batched_decorator", "unbatched_decorator"])
def simulator(request):
    if request.param == "class":
        simulator = Simulator()
    elif request.param == "batched_decorator":
        simulator = batched_decorated_simulator
    elif request.param == "unbatched_decorator":
        simulator = unbatched_decorated_simulator
    else:
        raise NotImplementedError

    return simulator


@pytest.fixture(params=[True, False])
def use_multiprocessing(request):
    return request.param


@pytest.fixture(params=[1, 2])
def workers(request):
    return request.param
