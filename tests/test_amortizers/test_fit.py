
import keras
import pytest

from tests.utils import InterruptFitCallback, FitInterruptedError
from .fixtures import *


def test_compile(amortizer):
    amortizer.compile(optimizer="AdamW")


def test_fit(amortizer, dataset):
    # TODO: verify the model learns something by comparing some metric before and after training
    amortizer.compile(optimizer="AdamW")
    amortizer.fit(dataset, epochs=2)


def test_interrupt_and_resume_fit(tmp_path, amortizer, dataset):
    # TODO: test the InterruptFitCallback
    amortizer.compile(optimizer="AdamW")

    callbacks = [
        InterruptFitCallback(epochs=1, error_type=FitInterruptedError),
        keras.callbacks.ModelCheckpoint(tmp_path / "model.keras"),
    ]

    with pytest.raises(FitInterruptedError):
        # interrupted fit (epochs 0-1)
        amortizer.fit(dataset, epochs=2, callbacks=callbacks)

    assert (tmp_path / "model.keras").exists(), "checkpoint has not been created"

    loaded_amortizer = keras.saving.load_model(tmp_path / "model.keras")

    # TODO: verify the fit is actually resumed (and not just started new with the existing weights)
    #  might require test code change
    # resume fit (epochs 1-2)
    loaded_amortizer.fit(dataset, epochs=2)


def test_extended_fit(amortizer, dataset):
    # TODO: verify that the model state is used to actually resume the fit
    #  might require test code change
    amortizer.compile(optimizer="AdamW")

    # initial fit (epochs 0-2)
    amortizer.fit(dataset, epochs=2)

    # extended fit (epochs 2-4)
    amortizer.fit(dataset, epochs=2)
