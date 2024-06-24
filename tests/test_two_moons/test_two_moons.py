import keras
import pytest

from tests.utils import assert_models_equal, max_mean_discrepancy
from tests.utils import InterruptFitCallback, FitInterruptedError


@pytest.mark.parametrize("jit_compile", [False, True])
def test_compile(approximator, random_samples, jit_compile):
    approximator.compile(jit_compile=jit_compile)


@pytest.mark.parametrize("jit_compile", [False, True])
def test_fit(approximator, train_dataset, validation_dataset, test_dataset, jit_compile):
    # TODO: Refactor to use approximator.sample() when implemented (instead of calling the inference network directly)

    approximator.compile(jit_compile=jit_compile, loss=keras.losses.KLDivergence())
    inf_vars = approximator.configurator.configure_inference_variables(test_dataset.data)
    inf_conds = approximator.configurator.configure_inference_conditions(test_dataset.data)
    y = test_dataset.data["x"]

    pre_loss = approximator.compute_metrics(train_dataset.data)["loss"]
    pre_val_loss = approximator.compute_metrics(validation_dataset.data)["loss"]
    x_before = approximator.inference_network(inf_vars, conditions=inf_conds)
    mmd_before = max_mean_discrepancy(x_before, y)

    history = approximator.fit(
        train_dataset,
        validation_data=validation_dataset,
        epochs=3,
    ).history
    x_after = approximator.inference_network(inf_vars, conditions=inf_conds)
    mmd_after = max_mean_discrepancy(x_after, y)

    # Test model weights have not vanished
    for layer in approximator.layers:
        for weight in layer.weights:
            assert not keras.ops.any(keras.ops.isnan(weight)).numpy()

    # Test KLD loss and validation loss decrease after training
    assert history["loss"][-1] < pre_loss
    assert history["val_loss"][-1] < pre_val_loss

    # Test MMD improved after training
    assert mmd_after < mmd_before


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
