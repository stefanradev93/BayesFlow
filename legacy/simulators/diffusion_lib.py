from numba import jit, prange
import ctypes
from numba.extending import get_cython_function_address
import numpy as np
import sys
import os


@jit(nopython=True, cache=True, parallel=True)
def simulate_diffusion_parallel(x, drifts, sv, zr, szr, a, ndt, sndt, alpha):
    """
    Simulate a dataset from the diffusion model.
    """

    # For each condition
    for k in prange(x.shape[1]):
    # For each trial
        for j in prange(x.shape[0]):
            x[j, k] = diffusion_trial(drifts[k], sv, zr, szr, a, ndt, sndt, alpha, 0.001, 5000)


def simulate_diffusion(drifts, zr=0.5, a=1.5, ndt=0.3, szr=0.0, sndt=0.0, alpha=2.0, sv=0.0, n_points=500, n_cond=2):
    """
    Simulates a diffusion process given a parameter vector.
    """

    x = np.zeros((n_points, n_cond), dtype=np.float32)
    simulate_diffusion_parallel(x, drifts, sv, zr, szr, a, ndt, sndt, alpha)
    return x


# Get a pointer to the C function diffusion.c
if __name__ != '__main__':

    # Add to path
    sys.path.append(sys.path.insert(0, os.path.join(
        os.path.dirname(os.path.dirname(os.path.realpath(__file__))), 'simulators')))

    addr_diffusion = get_cython_function_address("diffusion", "diffusion_trial")
    functype = ctypes.CFUNCTYPE(ctypes.c_double, ctypes.c_double, ctypes.c_double, 
                                ctypes.c_double, ctypes.c_double, ctypes.c_double,
                                ctypes.c_double, ctypes.c_double, ctypes.c_double, 
                                ctypes.c_double, ctypes.c_int)

    diffusion_trial = functype(addr_diffusion)