
import keras
import pytest

from tests.utils import InterruptFitCallback, FitInterruptedError


def test_fit(approximator, train_dataset, validation_dataset):
    # Test weights have not vainished and loss decreases after training
    approximator.compile(optimizer="AdamW")
    history = approximator.fit(
        x=train_dataset,
        validation_data=validation_dataset,
        epochs=2,
    ).history
    
    for layer in approximator.layers:
        for weight in layer.weights:
            assert not keras.ops.any(keras.ops.isnan(weight)).numpy()
    
    assert history["loss"][-1] < history["loss"][0]


@pytest.mark.skip(reason="not implemented")
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
