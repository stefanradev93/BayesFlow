import keras
import numpy as np
import pytest

from bayesflow.networks import DiffusionModel, FlowMatching, InferenceMasks, MaskHandler

from .conftest import make_network


def _numpy(value):
    return keras.ops.convert_to_numpy(value)


@pytest.mark.parametrize(
    ("key", "shape", "expected_shape"),
    [
        ("attention_mask", (3, 5), (3, 5)),
        ("attention_mask", (2, 3, 5), (8, 3, 5)),
        ("attention_mask", (1, 3, 5), (1, 3, 5)),
        ("mask", (3,), (3,)),
        ("mask", (2, 3), (8, 3)),
        ("mask", (1, 3), (1, 3)),
    ],
)
def test_mask_handler_repeats_only_batch_bearing_subnet_masks(key, shape, expected_shape):
    value = keras.ops.reshape(keras.ops.arange(np.prod(shape), dtype="float32"), shape)

    repeated = (
        MaskHandler()
        .repeat(
            InferenceMasks(subnet_kwargs={key: value}),
            num_repeats=4,
        )
        .subnet_kwargs[key]
    )

    assert tuple(repeated.shape) == expected_shape
    assert _numpy(repeated).dtype == np.float32
    if expected_shape[0] == 8:
        np.testing.assert_array_equal(_numpy(repeated), np.repeat(_numpy(value), 4, axis=0))


class InvariantRecordingSubnet(keras.Layer):
    """Zero field which records every solver state seen by the subnet."""

    def __init__(self, *, record=True):
        super().__init__()
        self.record = record
        self.states = []
        self.observed_condition_masks = []

    def call(
        self,
        inputs,
        training=None,
        fixed_target_mask=None,
        infer_target_mask=None,
        observed_condition_mask=None,
        attention_mask=None,
        **kwargs,
    ):
        if self.record:
            self.states.append(_numpy(inputs[0]).copy())
            if observed_condition_mask is not None:
                self.observed_condition_masks.append(_numpy(observed_condition_mask).copy())
        return keras.ops.zeros_like(inputs[0])

    def compute_output_shape(self, input_shape):
        return input_shape[0]


@pytest.mark.parametrize(
    ("network_name", "integrate_kwargs"),
    [
        ("diffusion", {"method": "langevin", "steps": 2, "step_size_factor": 0.001}),
        (
            "diffusion",
            {
                "method": "euler_maruyama",
                "steps": 2,
                "corrector_steps": 1,
                "step_size_factor": 0.001,
            },
        ),
        ("flow_matching", {"method": "glass", "steps": 2, "inner_steps": 1}),
    ],
    ids=["diffusion_langevin", "diffusion_corrector", "flow_matching_glass"],
)
def test_stochastic_solvers_preserve_fixed_state_at_every_subnet_call(
    network_name,
    integrate_kwargs,
    targets,
    conditions,
    fixed_target_mask,
):
    subnet = InvariantRecordingSubnet()
    network = make_network(
        network_name,
        subnet=subnet,
        integrate_kwargs=integrate_kwargs,
    )
    network.build(tuple(targets.shape), conditions_shape=tuple(conditions.shape))
    fixed_value = keras.ops.convert_to_tensor(
        np.asarray(
            [[-4.0, 7.5, 2.0], [3.0, -2.5, 1.0]],
            dtype=np.float32,
        )
    )

    samples = network.sample(
        (2,),
        conditions=conditions,
        seed=keras.random.SeedGenerator(91),
        fixed_target_mask=fixed_target_mask,
        fixed_target_value=fixed_value,
    )

    expected = _numpy(fixed_value)[:, 1]
    np.testing.assert_allclose(_numpy(samples)[:, 1], expected, rtol=0.0, atol=1e-6)
    assert subnet.states
    for state in subnet.states:
        np.testing.assert_allclose(state[:, 1], expected, rtol=0.0, atol=1e-6)


@pytest.mark.parametrize("use_jac", [False, True], ids=["direct", "jacobian"])
def test_compositional_diffusion_resolves_masks_once(
    monkeypatch,
    targets,
    conditions,
    fixed_target_mask,
    use_jac,
):
    subnet = InvariantRecordingSubnet(record=False)
    network = DiffusionModel(
        subnet=subnet,
        integrate_kwargs={"method": "euler", "steps": 2},
    )
    network.build(tuple(targets.shape), conditions_shape=tuple(conditions.shape))
    compositional_conditions = keras.ops.stack([conditions, -conditions], axis=1)

    original = network.mask_handler.resolve
    original_resolve_compositional = network.mask_handler.resolve_compositional
    original_without_masks = network.mask_handler.without_mask_kwargs
    original_repeat = network.mask_handler.repeat
    calls = {"resolve": 0, "resolve_compositional": 0, "without_masks": 0, "repeat": 0}

    def counting_resolve(*args, **kwargs):
        calls["resolve"] += 1
        return original(*args, **kwargs)

    def counting_resolve_compositional(*args, **kwargs):
        calls["resolve_compositional"] += 1
        return original_resolve_compositional(*args, **kwargs)

    def counting_without_masks(*args, **kwargs):
        calls["without_masks"] += 1
        return original_without_masks(*args, **kwargs)

    def counting_repeat(*args, **kwargs):
        calls["repeat"] += 1
        return original_repeat(*args, **kwargs)

    monkeypatch.setattr(network.mask_handler, "resolve", counting_resolve)
    monkeypatch.setattr(network.mask_handler, "resolve_compositional", counting_resolve_compositional)
    monkeypatch.setattr(network.mask_handler, "without_mask_kwargs", counting_without_masks)
    monkeypatch.setattr(network.mask_handler, "repeat", counting_repeat)

    samples = network._inverse_compositional(
        targets,
        conditions=compositional_conditions,
        compute_prior_score=lambda value, time: keras.ops.zeros_like(value),
        mini_batch_size=2,
        use_jac=use_jac,
        seed=keras.random.SeedGenerator(1729),
        fixed_target_mask=fixed_target_mask,
        fixed_target_value=targets,
    )

    assert tuple(samples.shape) == tuple(targets.shape)
    assert calls == {"resolve": 1, "resolve_compositional": 1, "without_masks": 1, "repeat": 1}


def test_learned_compositional_prior_is_marked_as_missing(targets, conditions):
    subnet = InvariantRecordingSubnet()
    network = DiffusionModel(subnet=subnet)
    network.build(tuple(targets.shape), conditions_shape=tuple(conditions.shape))
    compositional_conditions = keras.ops.stack([conditions, -conditions], axis=1)

    network.compositional_score(
        targets,
        time=0.5,
        conditions=compositional_conditions,
        seed=keras.random.SeedGenerator(1729),
        clip=None,
    )

    mask = subnet.observed_condition_masks[-1].reshape(2, 3, 2)
    np.testing.assert_array_equal(mask[:, :-1], np.ones((2, 2, 2), dtype=np.float32))
    np.testing.assert_array_equal(mask[:, -1], np.zeros((2, 2), dtype=np.float32))


def test_compositional_score_projects_fixed_coordinates(targets, conditions, fixed_target_mask):
    network = DiffusionModel(subnet=InvariantRecordingSubnet(record=False))
    network.build(tuple(targets.shape), conditions_shape=tuple(conditions.shape))
    compositional_conditions = keras.ops.stack([conditions, -conditions], axis=1)

    score = network.compositional_score(
        targets,
        time=0.5,
        conditions=compositional_conditions,
        seed=keras.random.SeedGenerator(1730),
        compute_prior_score=lambda value, time: keras.ops.ones_like(value),
        fixed_target_mask=fixed_target_mask,
        clip=None,
    )

    np.testing.assert_array_equal(_numpy(score)[:, 1], np.zeros((2,), dtype=np.float32))


@pytest.mark.skip_on_mps
def test_stable_consistency_all_fixed_loss_stays_zero_with_nonzero_weight_network(
    targets,
    conditions,
):
    network = make_network("stable_consistency")
    network.build(tuple(targets.shape), conditions_shape=tuple(conditions.shape))
    for weight in network.weight_fn_projector.weights:
        weight.assign(keras.ops.ones_like(weight) * keras.ops.cast(0.5, weight.dtype))

    metrics = network.compute_metrics(
        targets,
        conditions=conditions,
        fixed_target_mask=keras.ops.zeros_like(targets),
        stage="validation",
    )

    np.testing.assert_allclose(
        _numpy(metrics["loss"]),
        np.asarray(0.0, dtype=np.float32),
        rtol=0.0,
        atol=1e-6,
    )


def test_flow_matching_rejects_explicit_masks_with_optimal_transport(
    targets,
    conditions,
    fixed_target_mask,
):
    network = FlowMatching(
        use_optimal_transport=True,
        subnet_kwargs={"widths": (8, 8)},
        integrate_kwargs={"method": "euler", "steps": 2},
    )
    network.build(tuple(targets.shape), conditions_shape=tuple(conditions.shape))

    with pytest.raises(ValueError, match="optimal transport"):
        network.compute_metrics(
            targets,
            conditions=conditions,
            fixed_target_mask=fixed_target_mask,
            stage="training",
        )
