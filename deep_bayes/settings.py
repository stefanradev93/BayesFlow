EVIDENTIAL_BF = {
    'net_type'         :  "invariant",
    'n_models'         :  2,
    'learnable_pooling': True,
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
    'learnable_pooling': False,
    'dense_inv_args'   :  dict(units=64, activation='elu', kernel_initializer='glorot_normal'),
    'dense_equiv_args' :  dict(units=32, activation='elu', kernel_initializer='glorot_normal'),
    'dense_post_args'  :  dict(units=64, activation='elu', kernel_initializer='glorot_normal'),
    'n_dense_inv'      :  2,
    'n_dense_equiv'    :  2,
    'n_dense_post'     :  2,
    'n_equiv'          :  2
}

EVIDENTIAL_GAUSSIAN = {
    'net_type'         :  "invariant",
    'n_models'         :  400,
    'learnable_pooling': True,
    'dense_inv_args'   :  dict(units=128, activation='elu', kernel_initializer='glorot_normal'),
    'dense_equiv_args' :  dict(units=32, activation='elu', kernel_initializer='glorot_normal'),
    'dense_post_args'  :  dict(units=128, activation='elu', kernel_initializer='glorot_normal'),
    'n_dense_inv'      :  2,
    'n_dense_equiv'    :  2,
    'n_dense_post'     :  1,
    'n_equiv'          :  2
}

# --- Diffusion models --- #
EVIDENTIAL_DIFFUSION = {
    'net_type'         : 'invariant',
    'n_models'         :  6,
    'dense_inv_args'   :  dict(units=128, activation='elu'),
    'dense_equiv_args' :  dict(units=64, activation='elu'),
    'dense_post_args'  :  dict(units=128, activation='elu'),
    'learnable_pooling': True,
    'n_dense_post'     :  3,
    'n_dense_inv'      :  2,
    'n_dense_equiv'    :  2,
    'n_equiv'          :  2
}

# --- Epidemiology models --- #
EVIDENTIAL_JUMP = {
    'net_type'         : 'sequence',
    'lstm_units'       :  64,
    'conv_meta'        : None,
    'n_models'         :  2,
    'dense_post_args'  :  dict(units=32, activation='elu', kernel_initializer='glorot_normal'),
    'n_dense_post'     :  2,
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
    'learnable_pooling': False,
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
        'learnable_pooling': False,
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
        'learnable_pooling': False,
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
        'learnable_pooling': False,
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
    'conv_meta'        : [dict(filters=64, kernel_size=5, strides=1, activation='elu', kernel_initializer='glorot_normal', padding='same'),
                          dict(filters=128, kernel_size=3, strides=1, activation='elu', kernel_initializer='glorot_normal', padding='same'),
                          dict(filters=128, kernel_size=3, strides=1, activation='elu', kernel_initializer='glorot_normal', padding='same'),
                          ],
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


EVIDENTIAL_HH = {
    'net_type'         : 'sequence',
    'lstm_units'       :  128,
    'conv_meta'        : [dict(filters=64, kernel_size=5, strides=2, activation='elu', kernel_initializer='glorot_normal'),
                          dict(filters=64, kernel_size=5, strides=3, activation='elu', kernel_initializer='glorot_normal'),
                          dict(filters=128, kernel_size=3, strides=2, activation='elu', kernel_initializer='glorot_normal'),
                          dict(filters=128, kernel_size=3, strides=3, activation='elu', kernel_initializer='glorot_normal'),
                          dict(filters=128, kernel_size=2, strides=2, activation='elu', kernel_initializer='glorot_normal')
                          ],
    'n_models'         :  3,
    'dense_post_args'  :  dict(units=256, activation='elu', kernel_initializer='glorot_normal'),
    'n_dense_post'     :  2,
}