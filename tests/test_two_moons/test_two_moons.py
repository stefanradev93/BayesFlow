import copy
import keras
import pytest


from tests.utils import assert_models_equal
from tests.utils import InterruptFitCallback, FitInterruptedError


@pytest.mark.parametrize("jit_compile", [False, True])
def test_compile(approximator, random_samples, jit_compile):
    approximator.compile(jit_compile=jit_compile)


def test_fit(approximator, train_dataset, validation_dataset, batch_size):
    from bayesflow.metrics import MaximumMeanDiscrepancy

    approximator.compile(inference_metrics=[keras.metrics.KLDivergence(), MaximumMeanDiscrepancy()])

    approximator.build_from_data(train_dataset[0])

    untrained_weights = copy.deepcopy(approximator.weights)
    untrained_metrics = approximator.evaluate(validation_dataset, return_dict=True)

    approximator.fit(dataset=train_dataset, epochs=20, batch_size=batch_size)

    trained_weights = approximator.weights
    trained_metrics = approximator.evaluate(validation_dataset, return_dict=True)

    # check weights have changed during training
    assert any([keras.ops.any(~keras.ops.isclose(u, t)) for u, t in zip(untrained_weights, trained_weights)])

    assert isinstance(untrained_metrics, dict)
    assert isinstance(trained_metrics, dict)

    # test loss decreases
    assert "loss" in untrained_metrics
    assert "loss" in trained_metrics
    assert untrained_metrics["loss"] > trained_metrics["loss"]

    # test kl divergence decreases
    assert "kl_divergence" in untrained_metrics
    assert "kl_divergence" in trained_metrics
    assert untrained_metrics["kl_divergence"] > trained_metrics["kl_divergence"]

    # test mmd decreases
    assert "maximum_mean_discrepancy" in untrained_metrics
    assert "maximum_mean_discrepancy" in trained_metrics
    assert untrained_metrics["maximum_mean_discrepancy"] > trained_metrics["maximum_mean_discrepancy"]


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
