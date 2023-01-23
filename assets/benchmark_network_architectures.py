COUPLING_SETTINGS_TEST = {
    "t_args": {
        "dense_args": dict(units=8, kernel_initializer="glorot_uniform", activation="elu"),
        "num_dense": 1,
        "spec_norm": True,
    },
    "s_args": {
        "dense_args": dict(units=16, kernel_initializer="glorot_uniform", activation="elu"),
        "num_dense": 1,
        "spec_norm": False,
    },
}

NETWORK_SETTINGS = {
    "gaussian_linear": {
        "posterior": {"num_params": 10, "num_coupling_layers": 2, "coupling_net_settings": COUPLING_SETTINGS_TEST},
        "likelihood": {"num_params": 10, "num_coupling_layers": 2, "coupling_net_settings": COUPLING_SETTINGS_TEST},
    },
    "gaussian_linear_uniform": {
        "posterior": {"num_params": 10, "num_coupling_layers": 2, "coupling_net_settings": COUPLING_SETTINGS_TEST},
        "likelihood": {"num_params": 10, "num_coupling_layers": 2, "coupling_net_settings": COUPLING_SETTINGS_TEST},
    },
    "slcp": {
        "posterior": {"num_params": 5, "num_coupling_layers": 2, "coupling_net_settings": COUPLING_SETTINGS_TEST},
        "likelihood": {"num_params": 8, "num_coupling_layers": 2, "coupling_net_settings": COUPLING_SETTINGS_TEST},
    },
    "slcp_distractors": {
        "posterior": {"num_params": 5, "num_coupling_layers": 2, "coupling_net_settings": COUPLING_SETTINGS_TEST},
        "likelihood": {"num_params": 100, "num_coupling_layers": 2, "coupling_net_settings": COUPLING_SETTINGS_TEST},
    },
    "bernoulli_glm": {
        "posterior": {"num_params": 10, "num_coupling_layers": 2, "coupling_net_settings": COUPLING_SETTINGS_TEST},
        "likelihood": {"num_params": 10, "num_coupling_layers": 2, "coupling_net_settings": COUPLING_SETTINGS_TEST},
    },
    "bernoulli_glm_raw": {
        "posterior": {"num_params": 10, "num_coupling_layers": 2, "coupling_net_settings": COUPLING_SETTINGS_TEST},
        "likelihood": {"num_params": 2, "num_coupling_layers": 2, "coupling_net_settings": COUPLING_SETTINGS_TEST},
    },
    "gaussian_mixture": {
        "posterior": {"num_params": 2, "num_coupling_layers": 2, "coupling_net_settings": COUPLING_SETTINGS_TEST},
        "likelihood": {"num_params": 2, "num_coupling_layers": 2, "coupling_net_settings": COUPLING_SETTINGS_TEST},
    },
    "two_moons": {
        "posterior": {"num_params": 2, "num_coupling_layers": 3, "coupling_net_settings": COUPLING_SETTINGS_TEST},
        "likelihood": {"num_params": 2, "num_coupling_layers": 2, "coupling_net_settings": COUPLING_SETTINGS_TEST},
    },
    "sir": {
        "posterior": {"num_params": 2, "num_coupling_layers": 2, "coupling_net_settings": COUPLING_SETTINGS_TEST},
        "likelihood": {"num_params": 10, "num_coupling_layers": 3, "coupling_net_settings": COUPLING_SETTINGS_TEST},
    },
    "lotka_volterra": {
        "posterior": {"num_params": 4, "num_coupling_layers": 2, "coupling_net_settings": COUPLING_SETTINGS_TEST},
        "likelihood": {"num_params": 20, "num_coupling_layers": 2, "coupling_net_settings": COUPLING_SETTINGS_TEST},
    },
}
