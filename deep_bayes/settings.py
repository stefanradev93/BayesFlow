INVARIANT_DEFAULTS = {
    'net_type'         :  "invariant",
    'dense_inv_args'   :  dict(units=64, activation='elu'),
    'dense_equiv_args' :  dict(units=32, activation='elu'),
    'dense_post_args'  :  dict(units=64, activation='elu'),
    'n_dense_inv'      :  2,
    'n_dense_equiv'    :  2,
    'n_dense_post'     :  2,
    'n_equiv'          :  2
}

INVARIANT_DIFFUSION = {
    'dense_inv_args'   :  dict(units=128, activation='elu'),
    'dense_equiv_args' :  dict(units=64, activation='elu'),
    'n_dense_inv'      :  2,
    'n_dense_equiv'    :  2,
    'n_equiv'          :  2
}

# --- Memory models --- #
EVIDENTIAL_MEMORY = {
    'net_type'         : 'invariant',
    'n_models'         :  3,
    'dense_inv_args'   :  dict(units=64, activation='elu'),
    'dense_equiv_args' :  dict(units=32, activation='elu'),
    'dense_post_args'  :  dict(units=128, activation='elu'),
    'n_dense_post'     :  3,
    'n_dense_inv'      :  2,
    'n_dense_equiv'    :  2,
    'n_equiv'          :  2
}

DROPOUT_MEMORY = {
    'summary_type'    : 'invariant',
    'dropout_rate'    : 0.1,
    'n_models'        : 3,
    'dense_post_args' : dict(units=128, activation='elu'),
    'n_dense_post'    : 3,
    'summary_meta': {
        'dense_inv_args'   :  dict(units=64, activation='elu'),
        'dense_equiv_args' :  dict(units=32, activation='elu'),
        'n_dense_inv'      :  2,
        'n_dense_equiv'    :  2,
        'n_equiv'          :  2
    }
}

SOFTMAX_MEMORY = {
    'summary_type'    : 'invariant',
    'n_models'        : 3,
    'dense_post_args' : dict(units=128, activation='elu'),
    'n_dense_post'    : 3,
    'summary_meta': {
        'dense_inv_args'   :  dict(units=64, activation='elu'),
        'dense_equiv_args' :  dict(units=32, activation='elu'),
        'n_dense_inv'      :  2,
        'n_dense_equiv'    :  2,
        'n_equiv'          :  2
    }
}

VAE_MEMORY = {
    'summary_type':  'invariant',
    'n_models': 3,
    'z_dim': 3,
    'n_dense_encoder': 3,
    'encoder_dense_args': dict(units=64, activation='elu'),
    'summary_meta': {
        'dense_inv_args'   :  dict(units=64, activation='elu'),
        'dense_equiv_args' :  dict(units=32, activation='elu'),
        'n_dense_inv'      :  2,
        'n_dense_equiv'    :  2,
        'n_equiv'          :  2
    }
}

SEQUENCE_ECOLOGY = {
    'net_type'         :  "sequence",
    'lstm_units'       :  64,
    'dense_post_args'  :  dict(units=64, activation='elu'),
    'n_dense_post'     :  4,
    'batch_norm'       :  False
}

SEQUENCE_TUMOR = {
    'n_models'         : 3,
    'net_type'         :  "sequence",
    'lstm_units'       :  64,
    'dense_post_args'  :  dict(units=64, activation='elu'),
    'n_dense_post'     :  4
}


SEQUENCE_EPIDEMIOLOGY = {
    'n_models'         : 5,
    'net_type'         :  "sequence",
    'lstm_units'       :  64,
    'dense_post_args'  :  dict(units=64, activation='elu'),
    'n_dense_post'     :  4
}


INVERTIBLE_DEFAULTS = {
    'n_units': [64, 64, 64],
    'activation': 'elu',
    'w_decay': 0.00001,
    'initializer': 'glorot_uniform'
}
