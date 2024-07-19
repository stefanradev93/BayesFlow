from collections.abc import Mapping, Sequence
from itertools import chain
from keras.saving import (
    deserialize_keras_object as deserialize,
    register_keras_serializable as serializable,
    serialize_keras_object as serialize,
)

from bayesflow.types import Tensor

from .configurator import Configurator


DataT = Mapping[str, Tensor]
VarT = Sequence[Tensor | None]


@serializable(package="bayesflow.configurators")
class CompositeConfigurator(Configurator[DataT, VarT]):
    """Composes multiple configurators into a single configurator, sequentially."""

    def __init__(self, configurators: Sequence[Configurator[DataT, VarT]]):
        self.configurators = configurators
        self.variable_counts = None

    def configure(self, data: DataT) -> VarT:
        variables = [configurator.configure(data) for configurator in self.configurators]

        if self.variable_counts is None:
            # TODO: this is wrong, since variables is a list of lists
            self.variable_counts = [len(v) if v is not None else 0 for v in variables]

        return list(chain(*variables))

    def deconfigure(self, variables: VarT) -> DataT:
        data = {}
        start = 0
        for idx, configurator in enumerate(self.configurators):
            stop = start + self.variable_counts[idx]
            data |= configurator.deconfigure(variables[start:stop])
            start = stop

        return data

    @classmethod
    def from_config(cls, config: dict, custom_objects=None) -> "CompositeConfigurator":
        return cls([deserialize(configurator, custom_objects) for configurator in config.pop("configurators")])

    def get_config(self) -> dict:
        return {"configurators": [serialize(configurator) for configurator in self.configurators]}
