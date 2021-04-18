from numba import jit, prange
from tensorflow.keras import utils
import ctypes
from numba.extending import get_cython_function_address
import numpy as np

# Get a pointer to the C function levy.c
addr_levy = get_cython_function_address("levy", "levy_trial")
functype = ctypes.CFUNCTYPE(ctypes.c_double, ctypes.c_double, ctypes.c_double, 
                            ctypes.c_double, ctypes.c_double, ctypes.c_double,
                            ctypes.c_double, ctypes.c_double, ctypes.c_double, 
                            ctypes.c_double, ctypes.c_int)
levy_trial = functype(addr_levy)


@jit(nopython=True)
def levy_condition(v, params, n_obs, dt=0.001, max_steps=1e4):
    """A wrapper over the jit function."""

    x = np.empty((params.shape[0], n_obs))
    _levy_condition(v, params, x, dt, max_steps)
    return x

@jit(nopython=True, parallel=True)
def _levy_condition(v, params, x, dt, max_steps):
    """
    Simulate a batch from the diffusion model.
    ----------
    INPUT:
    v      - np.array of shape (n_batch, )
    params - np.array of shape (n_batch, n_shared_params):
        param index 0 - a
        param index 1 - zr
        param index 2 - t0
        param index 3 - alpha
    x      - np.array of shape (n_batch, n_obs) - zero padded arrays
    """

    # For each batch
    for i in prange(x.shape[0]):
        # For each trial
        for j in prange(x.shape[1]):
            x[i, j] = levy_trial(v[i], 0., params[i, 1], 0., 
                                params[i, 0], params[i, 2], 0., 
                                params[i, 3], dt, max_steps)


def levy_simulator(params, n_obs, dt=0.001, max_steps=1e4):
    """
    Simulates a levy process for 4 conditions with 8 parameters (a, zr, t0, alpha, v1, v2, v3, v4).
    """

    n1 = n2 = n3 = n_obs // 4
    n4 = n_obs - 3 * n1

    v1 = params[:, 4]
    v2 = params[:, 5]
    v3 = params[:, 6]
    v4 = params[:, 7]
    
    rt_c1 = levy_condition(v1, params, n1, dt=dt, max_steps=max_steps)
    rt_c2 = levy_condition(v2, params, n2, dt=dt, max_steps=max_steps)
    rt_c3 = levy_condition(v3, params, n3, dt=dt, max_steps=max_steps)
    rt_c4 = levy_condition(v4, params, n4, dt=dt, max_steps=max_steps)

    # Store rts
    rts = np.concatenate([rt_c1, rt_c2, rt_c3, rt_c4], axis=1)

    # Create conditions array and one-hot-encode it
    ind = np.stack(params.shape[0] * [np.concatenate((np.zeros(n1), np.ones(n2), 2*np.ones(n3), 3*np.ones(n3)))]).astype(np.int32)
    oh = utils.to_categorical()

    sim_data = np.stack((rts, ind), axis=-1)
    return sim_data
