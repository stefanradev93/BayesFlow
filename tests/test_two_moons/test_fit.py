
import keras
import pytest
import numpy as np

from tests.utils import InterruptFitCallback, FitInterruptedError


def test_simulator(simulator):
    # Test for randomness between data points
    data = simulator.sample((10,))
    assert len(np.unique(data["r"].numpy())) > 1
    assert len(np.unique(data["alpha"].numpy())) > 1
    assert len(np.unique(data["theta"].numpy())) > 1
    assert len(np.unique(data["x"].numpy())) > 1
    
    # Test data points follow formula: f(r,alpha,theta) = x
    for r, alpha, theta, x in zip(data["r"], data["alpha"], data["theta"], data["x"]):
        x1 = -keras.ops.abs(theta[..., :1] + theta[..., 1:]) / keras.ops.sqrt(2.0) + r * keras.ops.cos(alpha) + 0.25
        x2 = (-theta[..., :1] + theta[..., 1:]) / keras.ops.sqrt(2.0) + r * keras.ops.sin(alpha)
        assert x1.numpy()[0] == x[0].numpy()
        assert x2.numpy()[0] == x[1].numpy()


@pytest.mark.skip(reason="WIP")
def test_fit(approximator, train_dataset, validation_dataset, simulator):
    # print(simulator.sample((2,)))
    # print(simulator.sample((2,)))
    # print(simulator.sample((2,)))
    
    # Test weights have not vainished and loss decreases after training
    approximator.compile(optimizer="AdamW")
    history = approximator.fit(
        x=train_dataset,
        validation_data=validation_dataset,
        epochs=5,
    ).history
        
    data = train_dataset.data.copy()

    inf_vars = approximator.configurator.configure_inference_variables(data)
    inf_conds = approximator.configurator.configure_inference_conditions(data)
    
    z = approximator.inference_network(inf_vars, conditions=inf_conds)
    
    # print(z.numpy()[:100, :])

    # print("r:", keras.ops.mean(train_dataset.data["r"]))
    # print("a:", keras.ops.mean(train_dataset.data["alpha"]))
    # print("0:", keras.ops.mean(train_dataset.data["theta"], axis=0))
    # print("x:", keras.ops.mean(train_dataset.data["x"], axis=0))
    
    # print([key for key in train_dataset.data.keys()])
    
    
        
    # for layer in approximator.layers:
    #     for weight in layer.weights:
    #         assert not keras.ops.any(keras.ops.isnan(weight)).numpy()
    
    # assert history["loss"][-1] < history["loss"][0]


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
