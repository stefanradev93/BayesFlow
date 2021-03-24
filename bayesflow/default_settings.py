from abc import ABC


class Setting(ABC):
    """
    Abstract Base class for settings. It's here to potentially extend the setting functionality in future.
    """
    def __init__(self):
        pass


class MetaDictSetting(Setting):
    """
    Implements an interface for a default meta_dict with optional mandatory fields
    """
    def __init__(self, meta_dict: dict, mandatory_fields: list = []):
        self.meta_dict = meta_dict
        self.mandatory_fields = mandatory_fields


DEFAULT_SETTING_INVARIANT_NET = MetaDictSetting(
    meta_dict={
        'n_dense_s1': 2,
        'n_dense_s2': 2,
        'n_dense_s3': 2,
        'n_equiv':    2,
        'dense_s1_args': {'activation': 'relu', 'units': 32},
        'dense_s2_args': {'activation': 'relu', 'units': 64},
        'dense_s3_args': {'activation': 'relu', 'units': 32}
    },
    mandatory_fields=[]
)


DEFAULT_SETTING_INVERTIBLE_NET = MetaDictSetting(
    meta_dict={
        'n_coupling_layers': 4,
        's_args': {
            'units': [128, 128],
            'activation': 'elu',
            'initializer': 'glorot_uniform',
        },
        't_args': {
            'units': [128, 128],
            'activation': 'elu',
            'initializer': 'glorot_uniform',
        },
        'alpha': 1.9,
        'permute': True
    },
    mandatory_fields=["n_params"]
)