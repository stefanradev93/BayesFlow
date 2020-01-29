EVIDENTIAL_BF = {
    'net_type'         :  "invariant",
    'n_models'         :  2,
    'dense_inv_args'   :  dict(units=64, activation='elu', kernel_initializer='glorot_normal'),
    'dense_equiv_args' :  dict(units=32, activation='elu', kernel_initializer='glorot_normal'),
    'dense_post_args'  :  dict(units=64, activation='elu', kernel_initializer='glorot_normal'),
    'n_dense_inv'      :  2,
    'n_dense_equiv'    :  2,
    'n_dense_post'     :  2,
    'n_equiv'          :  2
}


EVIDENTIAL_DEFAULTS = {
    'net_type'         :  "invariant",
    'n_models'         :  2,
    'dense_inv_args'   :  dict(units=64, activation='elu', kernel_initializer='glorot_normal'),
    'dense_equiv_args' :  dict(units=32, activation='elu', kernel_initializer='glorot_normal'),
    'dense_post_args'  :  dict(units=64, activation='elu', kernel_initializer='glorot_normal'),
    'n_dense_inv'      :  2,
    'n_dense_equiv'    :  2,
    'n_dense_post'     :  2,
    'n_equiv'          :  2
}

# --- Diffusion models --- #
EVIDENTIAL_DIFFUSION = {
    'net_type'         : 'invariant',
    'n_models'         :  6,
    'dense_inv_args'   :  dict(units=128, activation='elu'),
    'dense_equiv_args' :  dict(units=64, activation='elu'),
    'dense_post_args'  :  dict(units=128, activation='elu'),
    'n_dense_post'     :  3,
    'n_dense_inv'      :  2,
    'n_dense_equiv'    :  2,
    'n_equiv'          :  2
}

DROPOUT_DIFFUSION = {
    'summary_type'    : 'invariant',
    'dropout_rate'    : 0.1,
    'n_models'        : 6,
    'dense_post_args' : dict(units=128, activation='elu'),
    'n_dense_post'    : 3,
    'summary_meta': {
        'dense_inv_args'   :  dict(units=128, activation='elu'),
        'dense_equiv_args' :  dict(units=64, activation='elu'),
        'n_dense_inv'      :  2,
        'n_dense_equiv'    :  2,
        'n_equiv'          :  2
    }
}

SOFTMAX_DIFFUSION = {
    'summary_type'    : 'invariant',
    'n_models'        : 6,
    'dense_post_args' : dict(units=128, activation='elu'),
    'n_dense_post'    : 3,
    'summary_meta': {
        'dense_inv_args'   :  dict(units=128, activation='elu'),
        'dense_equiv_args' :  dict(units=64, activation='elu'),
        'n_dense_inv'      :  2,
        'n_dense_equiv'    :  2,
        'n_equiv'          :  2
    }
}

VAE_DIFFUSION = {
    'summary_type':  'invariant',
    'n_models': 6,
    'z_dim': 6,
    'n_dense_encoder': 3,
    'encoder_dense_args': dict(units=128, activation='elu'),
    'summary_meta': {
        'dense_inv_args'   :  dict(units=128, activation='elu'),
        'dense_equiv_args' :  dict(units=64, activation='elu'),
        'n_dense_inv'      :  2,
        'n_dense_equiv'    :  2,
        'n_equiv'          :  2
    }
}

# --- Memory models --- #
EVIDENTIAL_MEMORY = {
    'net_type'         : 'invariant',
    'n_models'         :  3,
    'dense_inv_args'   :  dict(units=64, activation='elu', kernel_initializer='glorot_normal'),
    'dense_equiv_args' :  dict(units=32, activation='elu', kernel_initializer='glorot_normal'),
    'dense_post_args'  :  dict(units=128, activation='elu', kernel_initializer='glorot_normal'),
    'n_dense_post'     :  3,
    'n_dense_inv'      :  2,
    'n_dense_equiv'    :  2,
    'n_equiv'          :  2
}

DROPOUT_MEMORY = {
    'summary_type'    : 'invariant',
    'dropout_rate'    : 0.1,
    'n_models'        : 3,
    'dense_post_args' : dict(units=128, activation='elu', kernel_initializer='glorot_normal'),
    'n_dense_post'    : 3,
    'summary_meta': {
        'dense_inv_args'   :  dict(units=64, activation='elu', kernel_initializer='glorot_normal'),
        'dense_equiv_args' :  dict(units=32, activation='elu', kernel_initializer='glorot_normal'),
        'n_dense_inv'      :  2,
        'n_dense_equiv'    :  2,
        'n_equiv'          :  2
    }
}

SOFTMAX_MEMORY = {
    'summary_type'    : 'invariant',
    'n_models'        : 3,
    'dense_post_args' : dict(units=128, activation='elu', kernel_initializer='glorot_normal'),
    'n_dense_post'    : 3,
    'summary_meta': {
        'dense_inv_args'   :  dict(units=64, activation='elu', kernel_initializer='glorot_normal'),
        'dense_equiv_args' :  dict(units=32, activation='elu', kernel_initializer='glorot_normal'),
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
    'encoder_dense_args': dict(units=64, activation='elu', kernel_initializer='glorot_normal'),
    'summary_meta': {
        'dense_inv_args'   :  dict(units=64, activation='elu', kernel_initializer='glorot_normal'),
        'dense_equiv_args' :  dict(units=32, activation='elu', kernel_initializer='glorot_normal'),
        'n_dense_inv'      :  2,
        'n_dense_equiv'    :  2,
        'n_equiv'          :  2
    }
}

# --- Tumor models --- #
EVIDENTIAL_TUMOR = {
    'net_type'         : 'sequence',
    'lstm_units'       :  64,
    'n_models'         :  3,
    'dense_post_args'  :  dict(units=128, activation='elu', kernel_initializer='glorot_normal'),
    'n_dense_post'     :  3
}

DROPOUT_TUMOR = {
    'summary_type'    : 'sequence',
    'dropout_rate'    : 0.1,
    'n_models'        : 3,
    'dense_post_args' : dict(units=128, activation='elu', kernel_initializer='glorot_normal'),
    'n_dense_post'    : 3,
    'summary_meta': {
        'lstm_units' :  64,
    }
}

SOFTMAX_TUMOR = {
    'summary_type'    : 'sequence',
    'n_models'        : 3,
    'dense_post_args' : dict(units=128, activation='elu', kernel_initializer='glorot_normal'),
    'n_dense_post'    : 3,
    'summary_meta': {
        'lstm_units' :  64,
    }
}

VAE_TUMOR = {
    'summary_type':  'sequence',
    'n_models': 3,
    'z_dim': 3,
    'n_dense_encoder': 3,
    'encoder_dense_args': dict(units=128, activation='elu', kernel_initializer='glorot_normal'),
    'summary_meta': {
        'lstm_units' :  64,
    }
}

# --- Epidemiology models --- #
EVIDENTIAL_EPI = {
    'net_type'         : 'sequence',
    'lstm_units'       :  64,
    'n_models'         :  5,
    'dense_post_args'  :  dict(units=128, activation='elu', kernel_initializer='glorot_normal'),
    'n_dense_post'     :  3,
}

DROPOUT_EPI = {
    'summary_type'    : 'sequence',
    'dropout_rate'    : 0.1,
    'n_models'        : 5,
    'dense_post_args' : dict(units=128, activation='elu', kernel_initializer='glorot_normal'),
    'n_dense_post'    : 3,
    'summary_meta': {
        'lstm_units' :  64,
    }
}

SOFTMAX_EPI= {
    'summary_type'    : 'sequence',
    'n_models'        : 5,
    'dense_post_args' : dict(units=128, activation='elu', kernel_initializer='glorot_normal'),
    'n_dense_post'    : 3,
    'summary_meta': {
        'lstm_units' :  64,
    }
}

VAE_EPI = {
    'summary_type':  'sequence',
    'n_models': 5,
    'z_dim': 5,
    'n_dense_encoder': 3,
    'encoder_dense_args': dict(units=128, activation='elu', kernel_initializer='glorot_normal'),
    'summary_meta': {
        'lstm_units' :  64,
    }
}

INVERTIBLE_DEFAULTS = {
    'n_units': [64, 64, 64],
    'activation': 'elu',
    'w_decay': 0.00001,
    'initializer': 'glorot_uniform'
}
