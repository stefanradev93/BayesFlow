COUPLING_SETTINGS_TEST = {
    't_args': {
        'dense_args': dict(units=32, kernel_initializer='glorot_uniform', activation='elu'),
        'n_dense': 1,
        'spec_norm': True
    },
    's_args': {
        'dense_args': dict(units=32, kernel_initializer='glorot_uniform', activation='elu'),
        'n_dense': 1,
        'spec_norm': False
    }
}

NETWORK_SETTINGS = {
    'gaussian_linear': {
        'posterior': {
            'n_params': 10,
            'n_coupling_layers': 2,
            'coupling_settings': COUPLING_SETTINGS_TEST
        },
        'likelihood': {
            'n_params': 10,
            'n_coupling_layers': 2,
            'coupling_settings': COUPLING_SETTINGS_TEST
        }
    },
    'gaussian_linear_uniform': {
        'posterior': {
            'n_params': 10,
            'n_coupling_layers': 2,
            'coupling_settings': COUPLING_SETTINGS_TEST
        },
        'likelihood': {
            'n_params': 10,
            'n_coupling_layers': 2,
            'coupling_settings': COUPLING_SETTINGS_TEST
        }
    },
    'slcp': {
        'posterior': {
            'n_params': 5,
            'n_coupling_layers': 2,
            'coupling_settings': COUPLING_SETTINGS_TEST
        },
        'likelihood': {
            'n_params': 8,
            'n_coupling_layers': 2,
            'coupling_settings': COUPLING_SETTINGS_TEST
        }
    },
    'slcp_distractors': {
        'posterior': {
            'n_params': 5,
            'n_coupling_layers': 2,
            'coupling_settings': COUPLING_SETTINGS_TEST
        },
        'likelihood': {
            'n_params': 100, 
            'n_coupling_layers': 2,
            'coupling_settings': COUPLING_SETTINGS_TEST
        }
    },
    'bernoulli_glm': {
        'posterior': {
            'n_params': 10,
            'n_coupling_layers': 2,
            'coupling_settings': COUPLING_SETTINGS_TEST
        },
        'likelihood': {
            'n_params': 10,
            'n_coupling_layers': 2,
            'coupling_settings': COUPLING_SETTINGS_TEST
        }
    },
    'bernoulli_glm_raw': {
        'posterior': {
            'n_params': 10,
            'n_coupling_layers': 2,
            'coupling_settings': COUPLING_SETTINGS_TEST
        },
        'likelihood': {
            'n_params': 100,
            'n_coupling_layers': 2,
            'coupling_settings': COUPLING_SETTINGS_TEST
        }
    },
    'gaussian_mixture': {
        'posterior': {
            'n_params': 2,
            'n_coupling_layers': 2,
            'coupling_settings': COUPLING_SETTINGS_TEST
        },
        'likelihood': {
            'n_params': 2,
            'n_coupling_layers': 2,
            'coupling_settings': COUPLING_SETTINGS_TEST
        }
    },
    'two_moons': {
        'posterior': {
            'n_params': 2,
            'n_coupling_layers': 3,
            'coupling_settings': COUPLING_SETTINGS_TEST
        },
        'likelihood': {
            'n_params': 2,
            'n_coupling_layers': 2,
            'coupling_settings': COUPLING_SETTINGS_TEST
        }
    },
    'sir': {
        'posterior': {
            'n_params': 2,
            'n_coupling_layers': 2,
            'coupling_settings': COUPLING_SETTINGS_TEST
        },
        'likelihood': {
            'n_params': 10,
            'n_coupling_layers': 3,
            'coupling_settings': COUPLING_SETTINGS_TEST
        }
    },
    'lotka_volterra': {
        'posterior': {
            'n_params': 4,
            'n_coupling_layers': 2,
            'coupling_settings': COUPLING_SETTINGS_TEST
        },
        'likelihood': {
            'n_params': 20,
            'n_coupling_layers': 2,
            'coupling_settings': COUPLING_SETTINGS_TEST
        }
    }
}
