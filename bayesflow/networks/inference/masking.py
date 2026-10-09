"""Mask handling for inference networks."""

from __future__ import annotations

import inspect
from collections.abc import Callable, Iterable, Mapping
from dataclasses import dataclass, field, replace
from enum import StrEnum
from typing import Any

import keras

from bayesflow.types import Tensor
from bayesflow.utils.tensor_utils import repeat_and_flatten


class MaskName(StrEnum):
    """Canonical mask keyword names."""

    SUMMARY_ATTENTION = "summary_attention_mask"
    SUMMARY = "summary_mask"
    INFERENCE_ATTENTION = "inference_attention_mask"
    INFERENCE = "inference_mask"
    INFER_TARGET = "infer_target_mask"
    FIXED_TARGET = "fixed_target_mask"
    OBSERVED_CONDITION = "observed_condition_mask"
    FIXED_TARGET_VALUE = "fixed_target_value"
    ATTENTION_MASK = "attention_mask"
    MASK = "mask"

    @classmethod
    def semantic(cls) -> tuple[MaskName, ...]:
        return cls.INFER_TARGET, cls.FIXED_TARGET, cls.OBSERVED_CONDITION, cls.FIXED_TARGET_VALUE


@dataclass(frozen=True)
class InferenceMasks:
    """Masks prepared for one inference-network operation."""

    fixed_target_mask: Tensor | None = None
    infer_target_mask: Tensor | None = None
    observed_condition_mask: Tensor | None = None
    fixed_target_value: Tensor | None = None
    subnet_kwargs: Mapping[str, Tensor] = field(default_factory=dict)

    def as_kwargs(self) -> dict[str, Tensor]:
        return {name.value: value for name in MaskName.semantic() if (value := getattr(self, name.value)) is not None}


class MaskHandler:
    """Resolve semantic masks and route all mask kwargs."""

    summary_masks = {
        MaskName.SUMMARY_ATTENTION: MaskName.ATTENTION_MASK,
        MaskName.SUMMARY: MaskName.MASK,
    }
    inference_masks = {
        MaskName.INFERENCE_ATTENTION: MaskName.ATTENTION_MASK,
        MaskName.INFERENCE: MaskName.MASK,
    }
    subnet_masks = (
        MaskName.ATTENTION_MASK,
        MaskName.MASK,
        MaskName.FIXED_TARGET,
        MaskName.INFER_TARGET,
        MaskName.OBSERVED_CONDITION,
    )

    def __init__(
        self,
        *,
        fixed_target_prob: float = 0.0,
        missing_target_prob: float = 0.0,
        missing_conditions_prob: float = 0.0,
        seed_generator: keras.random.SeedGenerator | None = None,
    ):
        self.fixed_target_prob = fixed_target_prob
        self.missing_target_prob = missing_target_prob
        self.missing_conditions_prob = missing_conditions_prob
        self.seed_generator = seed_generator or keras.random.SeedGenerator()

    @staticmethod
    def collect(routes: Mapping[str, str] | Iterable[str], *sources: Mapping[str, Any]) -> dict[str, Any]:
        """Collect routed, non-``None`` values; later sources take precedence."""
        if isinstance(routes, Mapping):
            routes = {str(source): str(target) for source, target in routes.items()}
        else:
            routes = {str(name): str(name) for name in routes}

        return {
            target: source[source_name]
            for source in sources
            for source_name, target in routes.items()
            if source.get(source_name) is not None
        }

    @classmethod
    def collect_summary(cls, *sources: Mapping[str, Any]) -> dict[str, Any]:
        return cls.collect(cls.summary_masks, *sources)

    @classmethod
    def collect_inference(cls, *sources: Mapping[str, Any]) -> dict[str, Any]:
        return cls.collect(cls.inference_masks, *sources)

    @classmethod
    def collect_semantic(cls, *sources: Mapping[str, Any]) -> dict[str, Any]:
        return cls.collect(MaskName.semantic(), *sources)

    @classmethod
    def collect_masks(cls, *sources: Mapping[str, Any]) -> dict[str, Any]:
        return cls.collect(cls.subnet_masks, *sources) | cls.collect_semantic(*sources)

    @classmethod
    def collect_subnet_kwargs(cls, subnet: keras.Layer, *sources: Mapping[str, Any]) -> dict[str, Any]:
        """Collect masks explicitly declared by ``subnet.call``."""
        masks = cls.collect(cls.subnet_masks, *sources)
        parameters = inspect.signature(subnet.call).parameters
        return {name: value for name, value in masks.items() if name in parameters}

    @classmethod
    def without_semantic_kwargs(cls, source: Mapping[str, Any]) -> dict[str, Any]:
        names = {name.value for name in MaskName.semantic()}
        return {name: value for name, value in source.items() if name not in names}

    @classmethod
    def without_mask_kwargs(cls, source: Mapping[str, Any]) -> dict[str, Any]:
        names = {name.value for name in cls.subnet_masks} | {name.value for name in MaskName.semantic()}
        return {name: value for name, value in source.items() if name not in names}

    @staticmethod
    def maybe_apply(value: Tensor, mask: Tensor | None, replacement: Tensor | None = None) -> Tensor:
        """Apply ``mask`` to a tensor, optionally replacing masked entries."""
        if mask is None:
            return value
        if replacement is None:
            replacement = keras.ops.zeros_like(value)
        return keras.ops.where(keras.ops.cast(mask, "bool"), value, replacement)

    def resolve(
        self,
        *sources: Mapping[str, Any],
        targets: Tensor,
        conditions: Tensor | None = None,
        generate: bool = False,
        subnet: keras.Layer | None = None,
    ) -> InferenceMasks:
        """Resolve provided and sampled semantic masks once for an operation."""
        provided = self.collect_semantic(*sources)
        dtype = keras.config.floatx()
        subnet_parameters = None if subnet is None else inspect.signature(subnet.call).parameters

        fixed_target_mask = provided.get(MaskName.FIXED_TARGET)
        infer_target_mask = provided.get(MaskName.INFER_TARGET)

        if fixed_target_mask is None and generate and self.fixed_target_prob > 0.0:
            fixed_target_mask = self.random_mask(
                targets,
                drop_prob=self.fixed_target_prob,
                keep_one=True,
                eligible=infer_target_mask,
            )
        elif fixed_target_mask is not None:
            fixed_target_mask = keras.ops.broadcast_to(
                keras.ops.cast(fixed_target_mask, dtype),
                keras.ops.shape(targets),
            )

        if (
            infer_target_mask is None
            and fixed_target_mask is not None
            and generate
            and self.missing_target_prob > 0.0
            and (subnet_parameters is None or MaskName.INFER_TARGET in subnet_parameters)
        ):
            infer_target_mask = keras.ops.maximum(
                fixed_target_mask,
                self.random_mask(targets, drop_prob=self.missing_target_prob),
            )
        elif infer_target_mask is not None:
            infer_target_mask = keras.ops.broadcast_to(
                keras.ops.cast(infer_target_mask, dtype),
                keras.ops.shape(targets),
            )

        observed_condition_mask = provided.get(MaskName.OBSERVED_CONDITION)
        if (
            observed_condition_mask is None
            and generate
            and self.missing_conditions_prob > 0.0
            and (subnet_parameters is None or MaskName.OBSERVED_CONDITION in subnet_parameters)
        ):
            if conditions is not None:
                observed_condition_mask = self.random_mask(conditions, drop_prob=self.missing_conditions_prob)
        elif observed_condition_mask is not None and conditions is not None:
            observed_condition_mask = keras.ops.broadcast_to(
                keras.ops.cast(observed_condition_mask, dtype),
                keras.ops.shape(conditions),
            )

        fixed_target_value = provided.get(MaskName.FIXED_TARGET_VALUE)
        if fixed_target_value is not None:
            fixed_target_value = keras.ops.broadcast_to(
                keras.ops.cast(fixed_target_value, targets.dtype),
                keras.ops.shape(targets),
            )

        masks = InferenceMasks(
            fixed_target_mask=fixed_target_mask,
            infer_target_mask=infer_target_mask,
            observed_condition_mask=observed_condition_mask,
            fixed_target_value=fixed_target_value,
        )
        if subnet is None:
            return masks
        return replace(masks, subnet_kwargs=self.collect_subnet_kwargs(subnet, *sources, masks.as_kwargs()))

    def resolve_compositional(
        self,
        subnet: keras.Layer,
        *sources: Mapping[str, Any],
        targets: Tensor,
        conditions: Tensor,
    ) -> InferenceMasks:
        masks = self.resolve(*sources, targets=targets, conditions=conditions)
        if masks.observed_condition_mask is None:
            masks = replace(masks, observed_condition_mask=keras.ops.ones_like(conditions))
        return replace(masks, subnet_kwargs=self.collect_subnet_kwargs(subnet, *sources, masks.as_kwargs()))

    def repeat(self, masks: InferenceMasks, num_repeats: int) -> InferenceMasks:
        """Repeat canonical batched masks for flattened compositional inputs.

        ``mask`` is batched from rank 2 and ``attention_mask`` from rank 3. A
        leading singleton is already broadcastable and is left unchanged.
        """
        semantic_kwargs = self.collect_semantic(masks.subnet_kwargs)
        subnet_kwargs = {}
        for name, value in masks.subnet_kwargs.items():
            if name in semantic_kwargs:
                continue
            minimum_batch_rank = 3 if name == MaskName.ATTENTION_MASK else 2
            has_batch_axis = hasattr(value, "shape") and len(value.shape) >= minimum_batch_rank and value.shape[0] != 1
            subnet_kwargs[name] = repeat_and_flatten(value, num_repeats) if has_batch_axis else value

        repeated = {
            name: repeat_and_flatten(value, num_repeats)
            for name, value in masks.as_kwargs().items()
            if name != MaskName.OBSERVED_CONDITION
        }
        repeated_masks = replace(
            masks,
            **repeated,
            observed_condition_mask=None,
        )
        for name, value in repeated_masks.as_kwargs().items():
            if name in semantic_kwargs:
                subnet_kwargs[name] = value
        if MaskName.OBSERVED_CONDITION in semantic_kwargs:
            subnet_kwargs[MaskName.OBSERVED_CONDITION] = semantic_kwargs[MaskName.OBSERVED_CONDITION]
        return replace(repeated_masks, subnet_kwargs=subnet_kwargs)

    @staticmethod
    def with_observed_conditions(
        masks: InferenceMasks,
        observed_condition_mask: Tensor,
        num_repeats: int,
    ) -> InferenceMasks:
        """Flatten a compositional condition mask into prepared subnet masks."""
        condition_shape = tuple(keras.ops.shape(observed_condition_mask)[2:])
        observed_condition_mask = keras.ops.reshape(
            observed_condition_mask,
            (keras.ops.shape(observed_condition_mask)[0] * num_repeats,) + condition_shape,
        )
        subnet_kwargs = masks.subnet_kwargs
        if MaskName.OBSERVED_CONDITION in subnet_kwargs:
            subnet_kwargs = subnet_kwargs | {MaskName.OBSERVED_CONDITION: observed_condition_mask}
        return replace(
            masks,
            observed_condition_mask=observed_condition_mask,
            subnet_kwargs=subnet_kwargs,
        )

    def integrator_callbacks(
        self,
        masks: InferenceMasks,
        replacement: Tensor | None = None,
        state_key: str = "xz",
    ) -> tuple[Callable | None, Callable | None]:
        """Build fixed-target projections for deterministic and stochastic integrators."""
        if masks.fixed_target_mask is None:
            return None, None

        def project_state(state):
            fixed_value = masks.fixed_target_value if replacement is None else replacement
            return state | {
                state_key: self.maybe_apply(state[state_key], masks.fixed_target_mask, fixed_value),
            }

        def project_updates(state):
            return state | {state_key: self.maybe_apply(state[state_key], masks.fixed_target_mask)}

        return project_state, project_updates

    def random_mask(
        self,
        reference: Tensor,
        *,
        drop_prob: float,
        keep_one: bool = False,
        eligible: Tensor | None = None,
    ) -> Tensor:
        dtype = keras.config.floatx()

        random_values = keras.random.uniform(
            shape=keras.ops.shape(reference),
            dtype=dtype,
            seed=self.seed_generator,
        )
        mask = keras.ops.cast(random_values > drop_prob, dtype)

        if eligible is not None:
            eligible = keras.ops.cast(eligible, dtype)
            mask = mask * eligible

        if keep_one:
            candidates = random_values
            if eligible is not None:
                candidates = keras.ops.where(
                    keras.ops.cast(eligible, "bool"),
                    random_values,
                    keras.ops.full_like(random_values, -1.0),
                )
            keep = keras.ops.cast(candidates >= keras.ops.max(candidates, axis=-1, keepdims=True), dtype)
            if eligible is not None:
                has_eligible = keras.ops.any(keras.ops.cast(eligible, "bool"), axis=-1, keepdims=True)
                keep = keep * eligible * keras.ops.cast(has_eligible, dtype)
            mask = keras.ops.maximum(mask, keep)

        return mask
