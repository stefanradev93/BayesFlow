INVARIANT_DEFAULTS = {
    'dense_inv_args'   :  dict(units=64, activation='elu'),
    'dense_equiv_args' :  dict(units=32, activation='elu'),
    'n_dense_inv'      :  3,
    'n_dense_equiv'    :  3,
    'n_equiv'          :  2
}

INVARIANT_DIFFUSION = {
    'dense_inv_args'   :  dict(units=128, activation='elu'),
    'dense_equiv_args' :  dict(units=32, activation='elu'),
    'n_dense_inv'      :  3,
    'n_dense_equiv'    :  2,
    'n_equiv'          :  4
}


INVERTIBLE_DEFAULTS = {
    'n_units': [64, 64, 64],
    'activation': 'elu',
    'w_decay': 0.00001,
    'initializer': 'glorot_uniform'
}