
import keras
import pytest

from tests.utils import InterruptFitCallback, FitInterruptedError


def test_compile(amortizer):
    amortizer.compile(optimizer="AdamW")


def test_fit(amortizer, dataset):
    # TODO: verify the model learns something by comparing a metric before and after training
    amortizer.build((None, 2))
    amortizer.compile(optimizer="AdamW")
    amortizer.fit(dataset, epochs=10, steps_per_epoch=10, batch_size=32)


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
