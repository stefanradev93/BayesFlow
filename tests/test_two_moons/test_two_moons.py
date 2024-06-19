
import keras
import pytest

from tests.utils import assert_models_equal
from tests.utils import InterruptFitCallback, FitInterruptedError


@pytest.mark.parametrize("jit_compile", [False, True])
def test_compile(approximator, random_samples, jit_compile):
    approximator.compile(jit_compile=jit_compile)


@pytest.mark.parametrize("jit_compile", [False, True])
def test_fit(approximator, train_dataset, validation_dataset, jit_compile):
    # TODO: verify the model learns something by comparing a metric before and after training
    approximator.compile(jit_compile=jit_compile)
    approximator.fit(
        train_dataset,
        validation_data=validation_dataset,
        epochs=2,
    )


@pytest.mark.parametrize("jit_compile", [False, True])
def test_serialize_deserialize(tmp_path, approximator, random_samples, jit_compile):
    approximator.build_from_data(random_samples)

    keras.saving.save_model(approximator, tmp_path / "model.keras")
    loaded_approximator = keras.saving.load_model(tmp_path / "model.keras")

    assert_models_equal(approximator, loaded_approximator)



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
