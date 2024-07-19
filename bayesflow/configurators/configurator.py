from typing import Generic, TypeVar


DataT = TypeVar("DataT")
VarT = TypeVar("VarT")


class Configurator(Generic[DataT, VarT]):
    def configure(self, data: DataT) -> VarT:
        """Construct inference variables from data."""
        raise NotImplementedError

    def deconfigure(self, variables: VarT) -> DataT:
        """Reconstruct data from inference variables. Note that configuration is not required to be bijective, so this
        method is only meant to be a 'best effort' attempt, and may return incomplete or different data.
        """
        raise NotImplementedError

    @classmethod
    def from_config(cls, config: dict, custom_objects=None) -> "Configurator":
        """Construct a configurator from a configuration dictionary."""
        raise NotImplementedError

    def get_config(self) -> dict:
        """Return a configuration dictionary."""
        raise NotImplementedError
