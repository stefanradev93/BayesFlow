import inspect

import keras
import numpy as np

from bayesflow.adapters import Adapter
from bayesflow.approximators import ContinuousApproximator
from bayesflow.networks import FlowMatching


def _numpy(value):
    return keras.ops.convert_to_numpy(value)


class RecordingSubnet(keras.Layer):
    def __init__(self):
        super().__init__()
        self.attention_mask = None
        self.fixed_target_mask = None

    def call(
        self,
        inputs,
        training=None,
        attention_mask=None,
        fixed_target_mask=None,
        **kwargs,
    ):
        self.attention_mask = attention_mask
        self.fixed_target_mask = fixed_target_mask
        return inputs[0]

    def compute_output_shape(self, input_shape):
        return input_shape[0]


def _flow(subnet="time_mlp"):
    return FlowMatching(
        subnet=subnet,
        subnet_kwargs={"widths": (8, 8)},
        integrate_kwargs={"method": "euler", "steps": 2},
    )


def test_semantic_masks_are_not_named_approximator_parameters():
    parameters = inspect.signature(ContinuousApproximator.compute_metrics).parameters

    assert "fixed_target_mask" not in parameters
    assert "infer_target_mask" not in parameters
    assert "observed_condition_mask" not in parameters
    assert any(parameter.kind == inspect.Parameter.VAR_KEYWORD for parameter in parameters.values())


def test_compute_metrics_searches_semantic_kwargs_and_routes_attention(targets, conditions, fixed_target_mask):
    subnet = RecordingSubnet()
    network = _flow(subnet)
    approximator = ContinuousApproximator(inference_network=network, standardize=None)
    approximator.build(
        {
            "inference_variables": tuple(targets.shape),
            "inference_conditions": tuple(conditions.shape),
        }
    )
    attention = keras.ops.ones((2, 3, 5), dtype="float32")

    metrics = approximator.compute_metrics(
        targets,
        inference_conditions=conditions,
        inference_attention_mask=attention,
        fixed_target_mask=fixed_target_mask,
        stage="training",
    )

    assert np.isfinite(_numpy(metrics["loss"]))
    np.testing.assert_array_equal(_numpy(subnet.attention_mask), _numpy(attention))
    np.testing.assert_array_equal(_numpy(subnet.fixed_target_mask), _numpy(fixed_target_mask))


def test_default_adapter_preserves_canonical_semantic_mask_fields():
    adapter = ContinuousApproximator.build_adapter(
        inference_variables="theta",
        inference_conditions="x",
    )
    data = {
        "theta": np.asarray([[1.0, 2.0, 3.0]], dtype=np.float32),
        "x": np.asarray([[4.0, 5.0]], dtype=np.float32),
        "fixed_target_mask": np.asarray([[1.0, 0.0, 1.0]], dtype=np.float32),
        "fixed_target_value": np.asarray([[6.0, 7.0, 8.0]], dtype=np.float32),
        "infer_target_mask": np.asarray([[1.0, 1.0, 1.0]], dtype=np.float32),
        "observed_condition_mask": np.asarray([[1.0, 0.0]], dtype=np.float32),
    }

    adapted = adapter(data, strict=False, stage="inference")

    assert {
        "fixed_target_mask",
        "fixed_target_value",
        "infer_target_mask",
        "observed_condition_mask",
    }.issubset(adapted)
    for key in ("fixed_target_mask", "fixed_target_value", "infer_target_mask", "observed_condition_mask"):
        assert _numpy(adapted[key]).dtype == np.float32


def test_sample_discovers_masks_in_adapted_data_and_standardizes_fixed_values(targets, conditions):
    network = _flow()
    approximator = ContinuousApproximator(
        inference_network=network,
        adapter=Adapter(),
        standardize="inference_variables",
    )
    approximator.build(
        {
            "inference_variables": tuple(targets.shape),
            "inference_conditions": tuple(conditions.shape),
        }
    )
    approximator.compute_metrics(targets, inference_conditions=conditions, stage="training")

    fixed_mask = np.asarray([[1.0, 0.0, 1.0], [1.0, 0.0, 1.0]], dtype=np.float32)
    discovered_value = np.asarray([[-9.0, -9.0, -9.0], [-8.0, -8.0, -8.0]], dtype=np.float32)
    explicit_value = np.asarray([[4.0, 7.0, 2.0], [3.0, -2.0, 1.0]], dtype=np.float32)
    sample_conditions = {
        "inference_conditions": _numpy(conditions).astype(np.float32),
        "fixed_target_mask": fixed_mask,
        "fixed_target_value": discovered_value,
    }

    samples = approximator.sample(
        num_samples=1,
        conditions=sample_conditions,
        sample_shape=(),
        seed=keras.random.SeedGenerator(81),
        fixed_target_value=explicit_value,
    )["inference_variables"]

    assert samples.dtype == np.float32
    np.testing.assert_allclose(samples[:, 0, 1], explicit_value[:, 1], rtol=0.0, atol=1e-5)


def test_conditional_sampling_broadcasts_shared_semantic_masks(targets, conditions):
    network = _flow()
    approximator = ContinuousApproximator(
        inference_network=network,
        adapter=Adapter(),
        standardize=None,
    )
    approximator.build(
        {
            "inference_variables": tuple(targets.shape),
            "inference_conditions": tuple(conditions.shape),
        }
    )
    shared_mask = np.asarray([1.0, 0.0, 1.0], dtype=np.float32)
    shared_value = np.asarray([4.0, 7.0, 2.0], dtype=np.float32)

    samples = approximator.sample(
        num_samples=2,
        conditions={"inference_conditions": _numpy(conditions).astype(np.float32)},
        batch_size=1,
        sample_shape=(),
        seed=keras.random.SeedGenerator(82),
        fixed_target_mask=shared_mask,
        fixed_target_value=shared_value,
    )["inference_variables"]

    assert samples.dtype == np.float32
    np.testing.assert_allclose(
        samples[..., 1],
        np.full((2, 2), np.float32(7.0), dtype=np.float32),
        rtol=0.0,
        atol=1e-5,
    )


def test_unconditional_sampling_broadcasts_shared_fixed_values(targets):
    network = _flow()
    approximator = ContinuousApproximator(
        inference_network=network,
        adapter=Adapter(),
        standardize=None,
    )
    approximator.build({"inference_variables": tuple(targets.shape)})
    shared_mask = np.asarray([1.0, 0.0, 1.0], dtype=np.float32)
    shared_value = np.asarray([4.0, 7.0, 2.0], dtype=np.float32)

    samples = approximator.sample(
        num_samples=4,
        conditions=None,
        sample_shape=(),
        seed=keras.random.SeedGenerator(83),
        fixed_target_mask=shared_mask,
        fixed_target_value=shared_value,
    )["inference_variables"]

    assert samples.dtype == np.float32
    np.testing.assert_allclose(
        samples[:, 1],
        np.full((4,), np.float32(7.0), dtype=np.float32),
        rtol=0.0,
        atol=1e-5,
    )


def test_unconditional_sampling_discovers_batched_fixed_values(targets):
    network = _flow()
    approximator = ContinuousApproximator(
        inference_network=network,
        adapter=Adapter(),
        standardize=None,
    )
    approximator.build({"inference_variables": tuple(targets.shape)})
    fixed_mask = np.asarray([[1.0, 0.0, 1.0], [0.0, 1.0, 1.0]], dtype=np.float32)
    fixed_value = np.asarray([[4.0, 7.0, 2.0], [3.0, -2.0, 1.0]], dtype=np.float32)

    samples = approximator.sample(
        num_samples=3,
        conditions={
            "fixed_target_mask": fixed_mask,
            "fixed_target_value": fixed_value,
        },
        batch_size=1,
        sample_shape=(),
        seed=keras.random.SeedGenerator(84),
    )["inference_variables"]

    assert samples.shape == (2, 3, 3)
    assert samples.dtype == np.float32
    np.testing.assert_allclose(samples[0, :, 1], np.full((3,), np.float32(7.0)), rtol=0.0, atol=1e-5)
    np.testing.assert_allclose(samples[1, :, 0], np.full((3,), np.float32(3.0)), rtol=0.0, atol=1e-5)
