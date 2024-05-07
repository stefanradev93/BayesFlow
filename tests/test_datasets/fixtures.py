
import keras
import keras.random
import pytest

import bayesflow.experimental as bf


# TODO: do this last when the implementation of multiprocessing dataloading pipelines is done


@pytest.fixture()
def joint_distribution():
    class JointDistribution:
        def sample(self, batch_shape):
            return dict(x=keras.random.normal(batch_shape + (2,)))

        def log_prob(self, x):
            raise NotImplementedError()

    return JointDistribution()


@pytest.fixture()
def online_dataset(joint_distribution):
    return bf.datasets.OnlineDataset(joint_distribution)


@pytest.fixture()
def offline_dataset(tmp_path, joint_distribution):
    samples = joint_distribution.sample((32,))
    ...  # ?
    return bf.datasets.OfflineDataset(joint_distribution)
