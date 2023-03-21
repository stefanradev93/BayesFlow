COUPLING_SETTINGS_AFFINE = {
    "dense_args": dict(units=8, activation="elu"),
    "num_dense": 1,
}

COUPLING_SETTINGS_SPLINE = {"dense_args": dict(units=8, activation="elu"), "num_dense": 1, "bins": 4}

NETWORK_SETTINGS = {
    "gaussian_linear": {
        "posterior": {"num_params": 10, "num_coupling_layers": 2, "coupling_settings": COUPLING_SETTINGS_AFFINE},
        "likelihood": {"num_params": 10, "num_coupling_layers": 2, "coupling_settings": COUPLING_SETTINGS_AFFINE},
    },
    "gaussian_linear_uniform": {
        "posterior": {"num_params": 10, "num_coupling_layers": 2, "coupling_settings": COUPLING_SETTINGS_AFFINE},
        "likelihood": {
            "num_params": 10,
            "num_coupling_layers": 2,
            "coupling_settings": COUPLING_SETTINGS_SPLINE,
            "coupling_design": "spline",
        },
    },
    "slcp": {
        "posterior": {"num_params": 5, "num_coupling_layers": 2, "coupling_settings": COUPLING_SETTINGS_AFFINE},
        "likelihood": {
            "num_params": 8,
            "num_coupling_layers": 2,
            "coupling_settings": COUPLING_SETTINGS_SPLINE,
            "coupling_design": "spline",
        },
    },
    "slcp_distractors": {
        "posterior": {
            "num_params": 5,
            "num_coupling_layers": 2,
            "coupling_settings": COUPLING_SETTINGS_SPLINE,
            "coupling_design": "spline",
        },
        "likelihood": {
            "num_params": 100,
            "num_coupling_layers": 2,
            "coupling_settings": COUPLING_SETTINGS_SPLINE,
            "coupling_design": "spline",
        },
    },
    "bernoulli_glm": {
        "posterior": {"num_params": 10, "num_coupling_layers": 2, "coupling_settings": COUPLING_SETTINGS_AFFINE},
        "likelihood": {
            "num_params": 10,
            "num_coupling_layers": 2,
            "coupling_settings": COUPLING_SETTINGS_SPLINE,
            "coupling_design": "spline",
        },
    },
    "bernoulli_glm_raw": {
        "posterior": {
            "num_params": 10,
            "num_coupling_layers": 2,
            "coupling_settings": COUPLING_SETTINGS_SPLINE,
            "coupling_design": "spline",
        },
        "likelihood": {"num_params": 2, "num_coupling_layers": 2, "coupling_settings": COUPLING_SETTINGS_AFFINE},
    },
    "gaussian_mixture": {
        "posterior": {"num_params": 2, "num_coupling_layers": 2, "coupling_settings": COUPLING_SETTINGS_AFFINE},
        "likelihood": {
            "num_params": 2,
            "num_coupling_layers": 2,
            "coupling_settings": COUPLING_SETTINGS_SPLINE,
            "coupling_design": "spline",
        },
    },
    "two_moons": {
        "posterior": {"num_params": 2, "num_coupling_layers": 3, "coupling_settings": COUPLING_SETTINGS_AFFINE},
        "likelihood": {
            "num_params": 2,
            "num_coupling_layers": 2,
            "coupling_settings": COUPLING_SETTINGS_SPLINE,
            "coupling_design": "spline",
        },
    },
    "sir": {
        "posterior": {
            "num_params": 2,
            "num_coupling_layers": 2,
            "coupling_settings": COUPLING_SETTINGS_SPLINE,
            "coupling_design": "spline",
        },
        "likelihood": {"num_params": 10, "num_coupling_layers": 3, "coupling_settings": COUPLING_SETTINGS_AFFINE},
    },
    "lotka_volterra": {
        "posterior": {
            "num_params": 4,
            "num_coupling_layers": 2,
            "coupling_settings": COUPLING_SETTINGS_SPLINE,
            "coupling_design": "spline",
        },
        "likelihood": {
            "num_params": 20,
            "num_coupling_layers": 2,
            "coupling_settings": COUPLING_SETTINGS_SPLINE,
            "coupling_design": "spline",
        },
    },
    "inverse_kinematics": {
        "posterior": {"num_params": 4, "num_coupling_layers": 2, "coupling_settings": COUPLING_SETTINGS_AFFINE},
        "likelihood": {"num_params": 2, "num_coupling_layers": 2, "coupling_settings": COUPLING_SETTINGS_AFFINE},
    },
}
