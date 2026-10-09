import keras
import numpy as np
import pytest

from bayesflow.utils.serialization import deserialize, serialize


def _numpy(value):
    return keras.ops.convert_to_numpy(value)


def _build(network, targets, conditions):
    network.build(tuple(targets.shape), conditions_shape=tuple(conditions.shape))
    return network


def _skip_unsupported_torch_forward_ad(network_name, subnet):
    if network_name != "stable_consistency" or subnet != "diffusion_transformer":
        return
    if keras.backend.backend() != "torch":
        return

    import torch

    if torch.cuda.device_count() == 0:
        pytest.skip("PyTorch CPU attention does not support the forward-mode AD used by stable consistency.")


class RecordingSubnet(keras.Layer):
    """Minimal time subnet exposing every supported masking argument explicitly."""

    def __init__(self):
        super().__init__()
        self.received = {}

    def call(
        self,
        inputs,
        training=None,
        attention_mask=None,
        mask=None,
        fixed_target_mask=None,
        infer_target_mask=None,
        observed_condition_mask=None,
        **kwargs,
    ):
        self.received = {
            "state": inputs[0],
            "attention_mask": attention_mask,
            "mask": mask,
            "fixed_target_mask": fixed_target_mask,
            "infer_target_mask": infer_target_mask,
            "observed_condition_mask": observed_condition_mask,
        }
        return inputs[0]

    def compute_output_shape(self, input_shape):
        return input_shape[0]


def test_network_owns_handler_configuration(network_factory):
    network = network_factory(
        fixed_target_prob=0.2,
        missing_target_prob=0.3,
        missing_conditions_prob=0.4,
        seed=19,
    )

    assert network.mask_handler is not None
    assert network.mask_handler.seed_generator is network.seed_generator
    assert network.fixed_target_prob == pytest.approx(0.2)
    assert network.missing_target_prob == pytest.approx(0.3)
    assert network.missing_conditions_prob == pytest.approx(0.4)

    config = network.get_config()
    assert config["fixed_target_prob"] == pytest.approx(0.2)
    assert config["missing_target_prob"] == pytest.approx(0.3)
    assert config["missing_conditions_prob"] == pytest.approx(0.4)
    assert config["seed"] == 19


def test_mask_probabilities_remain_settable_for_custom_network_compatibility(network_factory):
    network = network_factory()

    network.fixed_target_prob = 0.2
    network.missing_target_prob = 0.3
    network.missing_conditions_prob = 0.4

    assert network.mask_handler.fixed_target_prob == pytest.approx(0.2)
    assert network.mask_handler.missing_target_prob == pytest.approx(0.3)
    assert network.mask_handler.missing_conditions_prob == pytest.approx(0.4)


def test_mask_configuration_survives_serialization(
    network_factory,
    targets,
    conditions,
):
    network = _build(
        network_factory(
            fixed_target_prob=0.2,
            missing_target_prob=0.3,
            missing_conditions_prob=0.4,
            seed=19,
        ),
        targets,
        conditions,
    )

    restored = deserialize(serialize(network))

    assert restored.fixed_target_prob == pytest.approx(0.2)
    assert restored.missing_target_prob == pytest.approx(0.3)
    assert restored.missing_conditions_prob == pytest.approx(0.4)
    assert restored.seed == 19
    assert restored.mask_handler.seed_generator is restored.seed_generator


@pytest.mark.parametrize("subnet", ["time_mlp", "diffusion_transformer"])
def test_probability_generated_masking_has_finite_training_loss(
    network_name,
    network_factory,
    subnet,
    targets,
    conditions,
):
    _skip_unsupported_torch_forward_ad(network_name, subnet)
    network = _build(
        network_factory(
            subnet=subnet,
            fixed_target_prob=0.5,
            missing_target_prob=0.5,
            missing_conditions_prob=0.5,
        ),
        targets,
        conditions,
    )

    metrics = network.compute_metrics(targets, conditions=conditions, stage="training")

    loss = _numpy(metrics["loss"])
    assert np.asarray(loss).dtype == np.float32
    assert np.all(np.isfinite(loss))


def test_explicit_masks_override_sampling_and_reach_explicit_subnet_parameters(
    network_factory,
    targets,
    conditions,
    fixed_target_mask,
    infer_target_mask,
    observed_condition_mask,
):
    subnet = RecordingSubnet()
    network = _build(
        network_factory(
            subnet=subnet,
            fixed_target_prob=1.0,
            missing_target_prob=1.0,
            missing_conditions_prob=1.0,
        ),
        targets,
        conditions,
    )
    attention_mask = keras.ops.ones((2, 3, 5), dtype="float32")
    padding_mask = keras.ops.ones((2, 3), dtype="float32")

    metrics = network.compute_metrics(
        targets,
        conditions=conditions,
        stage="training",
        attention_mask=attention_mask,
        mask=padding_mask,
        fixed_target_mask=fixed_target_mask,
        infer_target_mask=infer_target_mask,
        observed_condition_mask=observed_condition_mask,
    )

    assert np.isfinite(_numpy(metrics["loss"]))
    np.testing.assert_array_equal(_numpy(subnet.received["attention_mask"]), _numpy(attention_mask))
    np.testing.assert_array_equal(_numpy(subnet.received["mask"]), _numpy(padding_mask))
    np.testing.assert_array_equal(_numpy(subnet.received["fixed_target_mask"]), _numpy(fixed_target_mask))
    np.testing.assert_array_equal(_numpy(subnet.received["infer_target_mask"]), _numpy(infer_target_mask))
    np.testing.assert_array_equal(
        _numpy(subnet.received["observed_condition_mask"]),
        _numpy(observed_condition_mask),
    )


def test_all_fixed_targets_have_zero_training_loss(network_factory, targets, conditions):
    network = _build(network_factory(), targets, conditions)
    fixed_target_mask = keras.ops.zeros_like(targets)

    metrics = network.compute_metrics(
        targets,
        conditions=conditions,
        stage="training",
        fixed_target_mask=fixed_target_mask,
    )

    np.testing.assert_allclose(_numpy(metrics["loss"]), np.asarray(0.0, dtype=np.float32), atol=1e-6)


@pytest.mark.parametrize("subnet", ["time_mlp", "diffusion_transformer"])
def test_sampling_keeps_fixed_targets_exactly(
    network_name,
    network_factory,
    subnet,
    targets,
    conditions,
    fixed_target_mask,
):
    _skip_unsupported_torch_forward_ad(network_name, subnet)
    network = _build(network_factory(subnet=subnet), targets, conditions)
    fixed_target_value = keras.ops.convert_to_tensor(
        np.asarray(
            [
                [-4.0, 7.5, 2.0],
                [3.0, -2.5, 1.0],
            ],
            dtype=np.float32,
        )
    )

    samples = network.sample(
        (2,),
        conditions=conditions,
        seed=keras.random.SeedGenerator(314),
        fixed_target_mask=fixed_target_mask,
        fixed_target_value=fixed_target_value,
    )

    assert _numpy(samples).dtype == np.float32
    np.testing.assert_allclose(
        _numpy(samples)[:, 1],
        _numpy(fixed_target_value)[:, 1],
        rtol=0.0,
        atol=1e-6,
    )


def test_flow_matching_forward_uses_explicit_fixed_target_value(
    targets,
    conditions,
    fixed_target_mask,
):
    from bayesflow.networks import FlowMatching

    subnet = RecordingSubnet()
    network = FlowMatching(
        subnet=subnet,
        integrate_kwargs={"method": "euler", "steps": 2},
    )
    network.build(tuple(targets.shape), conditions_shape=tuple(conditions.shape))
    fixed_target_value = keras.ops.convert_to_tensor(
        np.asarray(
            [[-4.0, 7.5, 2.0], [3.0, -2.5, 1.0]],
            dtype=np.float32,
        )
    )

    network(
        targets,
        conditions=conditions,
        fixed_target_mask=fixed_target_mask,
        fixed_target_value=fixed_target_value,
    )

    np.testing.assert_allclose(
        _numpy(subnet.received["state"])[:, 1],
        _numpy(fixed_target_value)[:, 1],
        rtol=0.0,
        atol=1e-6,
    )
