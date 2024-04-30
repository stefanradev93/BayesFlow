
import keras
import keras.saving
import pytest

import bayesflow.experimental as bf

from tests.utils import *


# TODO:
#  current problems & TODOs:
#  - keras.utils.PyDataset does not exist under torch (jax?) backend ==> workaround?
#  - implement git workflow to test each backend
#  - running pytest ignores the backend env variable -

@pytest.fixture()
def inference_network():
    return keras.Sequential([
        keras.layers.Input(shape=(2,)),
        keras.layers.Dense(2),
    ])


@pytest.fixture()
def summary_network():
    return None


@pytest.fixture()
def amortizer(inference_network, summary_network):
    return bf.Amortizer(inference_network)


def test_fit(amortizer, dataset):
    # TODO: verify the model learns something?
    amortizer.fit(dataset, epochs=2)


def test_interrupt_and_resume_fit(tmp_path, amortizer, dataset):
    # TODO: check
    callbacks = [
        InterruptFitCallback(epochs=1),
        keras.callbacks.ModelCheckpoint(tmp_path / "model.keras"),
    ]

    with pytest.raises(RuntimeError):
        # interrupted fit
        amortizer.fit(dataset, epochs=2, callbacks=callbacks)

    assert (tmp_path / "model.keras").exists(), "checkpoint has not been created"

    loaded_amortizer = keras.saving.load_model(tmp_path / "model.keras")

    # TODO: verify the fit is actually resumed (and not just started new with the existing weights)
    # resume fit
    loaded_amortizer.fit(dataset, epochs=2)


def test_extended_fit(amortizer, dataset):
    # TODO: verify that the model state is used to actually resume the fit
    # initial fit
    amortizer.fit(dataset, epochs=2)

    # extended fit
    amortizer.fit(dataset, epochs=2)


def test_save_and_load(tmp_path, amortizer):
    amortizer.save(tmp_path / "amortizer.keras")
    loaded_amortizer = keras.saving.load_model(tmp_path / "amortizer.keras")

    assert_models_equal(amortizer, loaded_amortizer)
