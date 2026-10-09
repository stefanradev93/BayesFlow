import keras
import numpy as np

from bayesflow.networks import InferenceMasks, MaskHandler, MaskName


def _tensor(values):
    return keras.ops.convert_to_tensor(np.asarray(values, dtype=np.float32))


def _numpy(value):
    return keras.ops.convert_to_numpy(value)


def test_collect_inference_routes_names_and_uses_later_precedence():
    first = {"inference_attention_mask": _tensor([1.0, 0.0])}
    second = {"inference_attention_mask": _tensor([0.0, 1.0]), "inference_mask": None}

    collected = MaskHandler.collect_inference(first, second)

    assert set(collected) == {"attention_mask"}
    np.testing.assert_array_equal(_numpy(collected["attention_mask"]), np.asarray([0.0, 1.0], dtype=np.float32))


def test_collect_summary_routes_attention_and_padding_masks():
    attention = _tensor([1.0, 0.0])
    padding = _tensor([0.0, 1.0])

    collected = MaskHandler.collect_summary(
        {
            "summary_attention_mask": attention,
            "summary_mask": padding,
        }
    )

    assert set(collected) == {"attention_mask", "mask"}
    np.testing.assert_array_equal(_numpy(collected["attention_mask"]), _numpy(attention))
    np.testing.assert_array_equal(_numpy(collected["mask"]), _numpy(padding))


def test_collect_semantic_uses_later_precedence():
    discovered = {
        MaskName.FIXED_TARGET: _tensor([1.0, 0.0]),
        MaskName.FIXED_TARGET_VALUE: _tensor([2.0, 3.0]),
        MaskName.ATTENTION_MASK: _tensor([0.0, 1.0]),
        MaskName.MASK: _tensor([1.0, 1.0]),
    }
    explicit = {MaskName.FIXED_TARGET_VALUE: _tensor([4.0, 5.0])}

    collected = MaskHandler.collect_semantic(discovered, explicit)

    np.testing.assert_array_equal(
        _numpy(collected[MaskName.FIXED_TARGET_VALUE]),
        np.asarray([4.0, 5.0], dtype=np.float32),
    )
    assert set(collected) == {MaskName.FIXED_TARGET, MaskName.FIXED_TARGET_VALUE}


def test_inference_masks_expose_only_semantic_kwargs(fixed_target_mask):
    attention_mask = keras.ops.ones((2, 3, 3), dtype="float32")
    masks = InferenceMasks(
        fixed_target_mask=fixed_target_mask,
        subnet_kwargs={MaskName.ATTENTION_MASK: attention_mask},
    )

    assert set(masks.as_kwargs()) == {MaskName.FIXED_TARGET}


def test_collect_and_remove_all_mask_kwargs():
    source = {
        "attention_mask": _tensor([[1.0, 0.0]]),
        "mask": _tensor([1.0, 0.0]),
        MaskName.FIXED_TARGET: _tensor([0.0, 1.0]),
        MaskName.FIXED_TARGET_VALUE: _tensor([2.0, 3.0]),
        "steps": 4,
    }

    assert set(MaskHandler.collect_masks(source)) == {
        "attention_mask",
        "mask",
        MaskName.FIXED_TARGET,
        MaskName.FIXED_TARGET_VALUE,
    }
    assert MaskHandler.without_mask_kwargs(source) == {"steps": 4}


def test_resolve_does_not_generate_masks_unless_requested(targets, conditions):
    handler = MaskHandler(
        fixed_target_prob=1.0,
        missing_target_prob=1.0,
        missing_conditions_prob=1.0,
        seed_generator=keras.random.SeedGenerator(12),
    )

    masks = handler.resolve(targets=targets, conditions=conditions, generate=False)

    assert masks == InferenceMasks()


def test_resolve_generates_consistent_float32_masks(targets, conditions):
    handler = MaskHandler(
        fixed_target_prob=1.0,
        missing_target_prob=1.0,
        missing_conditions_prob=1.0,
        seed_generator=keras.random.SeedGenerator(21),
    )

    masks = handler.resolve(targets=targets, conditions=conditions, generate=True)
    fixed = _numpy(masks.fixed_target_mask)
    inferred = _numpy(masks.infer_target_mask)
    observed = _numpy(masks.observed_condition_mask)

    assert fixed.dtype == np.float32
    assert inferred.dtype == np.float32
    assert observed.dtype == np.float32
    assert np.all(fixed.sum(axis=-1) >= 1.0)
    assert np.all(inferred >= fixed)
    np.testing.assert_array_equal(observed, np.zeros_like(observed, dtype=np.float32))


def test_random_mask_is_a_public_handler_operation(targets):
    mask = MaskHandler(seed_generator=keras.random.SeedGenerator(22)).random_mask(
        targets,
        drop_prob=1.0,
        keep_one=True,
    )

    assert _numpy(mask).dtype == np.float32
    assert np.all(_numpy(mask).sum(axis=-1) >= 1.0)


def test_explicit_masks_override_generation(targets, conditions, fixed_target_mask):
    handler = MaskHandler(
        fixed_target_prob=1.0,
        missing_target_prob=1.0,
        missing_conditions_prob=1.0,
        seed_generator=keras.random.SeedGenerator(7),
    )
    infer = keras.ops.ones_like(targets)
    observed = keras.ops.ones_like(conditions)
    masks = handler.resolve(
        {
            MaskName.FIXED_TARGET: fixed_target_mask,
            MaskName.INFER_TARGET: infer,
            MaskName.OBSERVED_CONDITION: observed,
        },
        targets=targets,
        conditions=conditions,
        generate=True,
    )

    np.testing.assert_array_equal(_numpy(masks.fixed_target_mask), _numpy(fixed_target_mask))
    np.testing.assert_array_equal(_numpy(masks.infer_target_mask), _numpy(infer))
    np.testing.assert_array_equal(_numpy(masks.observed_condition_mask), _numpy(observed))


def test_maybe_apply_masks_and_replaces_values(targets, fixed_target_mask):
    handler = MaskHandler()
    fixed_value = keras.ops.full_like(targets, keras.ops.cast(9.0, targets.dtype))

    replaced = handler.maybe_apply(targets, fixed_target_mask, replacement=fixed_value)
    projected = handler.maybe_apply(keras.ops.ones_like(targets), fixed_target_mask)

    expected = _numpy(targets).copy()
    expected[:, 1] = np.float32(9.0)
    np.testing.assert_array_equal(_numpy(replaced), expected)
    np.testing.assert_array_equal(_numpy(projected), _numpy(fixed_target_mask))
    assert handler.maybe_apply(targets, None) is targets


def test_maybe_apply_isolates_nan_placeholders():
    handler = MaskHandler()
    values = _tensor([[2.0, np.nan, 4.0]])
    replacement = _tensor([[np.nan, 7.0, np.nan]])
    mask = _tensor([[1.0, 0.0, 1.0]])

    replaced = _numpy(handler.maybe_apply(values, mask, replacement))
    projected = _numpy(handler.maybe_apply(values, mask))

    np.testing.assert_array_equal(replaced, np.asarray([[2.0, 7.0, 4.0]], dtype=np.float32))
    np.testing.assert_array_equal(projected, np.asarray([[2.0, 0.0, 4.0]], dtype=np.float32))
    assert replaced.dtype == np.float32
    assert projected.dtype == np.float32


def test_subnet_capability_requires_explicit_parameters(fixed_target_mask):
    class ExplicitSubnet(keras.Layer):
        def call(self, inputs, fixed_target_mask=None, attention_mask=None, **kwargs):
            return inputs[0]

    class CatchAllSubnet(keras.Layer):
        def call(self, inputs, **kwargs):
            return inputs[0]

    attention = keras.ops.ones((2, 3, 3), dtype="float32")
    source = {
        MaskName.FIXED_TARGET: fixed_target_mask,
        "attention_mask": attention,
    }

    assert set(MaskHandler.collect_subnet_kwargs(ExplicitSubnet(), source)) == {
        "attention_mask",
        MaskName.FIXED_TARGET,
    }
    assert MaskHandler.collect_subnet_kwargs(CatchAllSubnet(), source) == {}


def test_resolve_routes_only_explicit_subnet_masks(targets, fixed_target_mask):
    class ExplicitSubnet(keras.Layer):
        def call(self, inputs, fixed_target_mask=None):
            return inputs[0]

    source = {
        MaskName.FIXED_TARGET: fixed_target_mask,
        MaskName.ATTENTION_MASK: keras.ops.ones((2, 3, 3), dtype="float32"),
    }
    masks = MaskHandler().resolve(source, targets=targets, subnet=ExplicitSubnet())

    assert set(masks.subnet_kwargs) == {MaskName.FIXED_TARGET}


def test_resolve_does_not_sample_masks_unsupported_by_subnet(targets, conditions):
    class CatchAllSubnet(keras.Layer):
        def call(self, inputs, **kwargs):
            return inputs[0]

    seed_generator = keras.random.SeedGenerator(23)
    handler = MaskHandler(
        missing_conditions_prob=1.0,
        seed_generator=seed_generator,
    )
    state_before = _numpy(seed_generator.state).copy()

    masks = handler.resolve(
        targets=targets,
        conditions=conditions,
        generate=True,
        subnet=CatchAllSubnet(),
    )

    assert masks.observed_condition_mask is None
    np.testing.assert_array_equal(_numpy(seed_generator.state), state_before)


def test_with_observed_conditions_flattens_and_routes_mask():
    observed = keras.ops.reshape(keras.ops.arange(12, dtype="float32"), (2, 3, 2))
    masks = InferenceMasks(subnet_kwargs={MaskName.OBSERVED_CONDITION: observed})

    updated = MaskHandler.with_observed_conditions(masks, observed, num_repeats=3)
    expected = np.asarray(observed).reshape(6, 2)

    np.testing.assert_array_equal(_numpy(updated.observed_condition_mask), expected)
    np.testing.assert_array_equal(_numpy(updated.subnet_kwargs[MaskName.OBSERVED_CONDITION]), expected)


def test_integrator_callbacks_project_state_and_updates(targets, fixed_target_mask):
    fixed_value = keras.ops.full_like(targets, keras.ops.cast(9.0, targets.dtype))
    masks = InferenceMasks(fixed_target_mask=fixed_target_mask, fixed_target_value=fixed_value)
    handler = MaskHandler()

    project_state, project_updates = handler.integrator_callbacks(masks)
    state = {"xz": targets, "other": keras.ops.ones((2,), dtype="float32")}
    projected_state = project_state(state)
    projected_updates = project_updates(state)

    expected_state = _numpy(targets).copy()
    expected_state[:, 1] = np.float32(9.0)
    expected_updates = _numpy(targets).copy()
    expected_updates[:, 1] = np.float32(0.0)
    np.testing.assert_array_equal(_numpy(projected_state["xz"]), expected_state)
    np.testing.assert_array_equal(_numpy(projected_updates["xz"]), expected_updates)
    assert projected_state["other"] is state["other"]
    assert handler.integrator_callbacks(InferenceMasks()) == (None, None)
