from libc.math cimport log, log1p, M_PI, tan, sqrt, sin, pow, cos, abs
from libc.stdlib cimport rand, RAND_MAX
cimport cython


@cython.nonecheck(False)
@cython.cdivision(True) 
cdef double random_uniform():
    cdef double r = rand()
    return r / RAND_MAX


@cython.nonecheck(False)
@cython.cdivision(True) 
cdef double random_exponential(double mu):
    
    cdef double u = random_uniform()
    return -mu * log1p(-u)


@cython.nonecheck(False)
@cython.cdivision(True) 
cdef double random_gaussian():
    cdef double x1, x2, w

    w = 2.0
    while (w >= 1.0):
        x1 = 2.0 * random_uniform() - 1.0
        x2 = 2.0 * random_uniform() - 1.0
        w = x1 * x1 + x2 * x2

    w = ((-2.0 * log(w)) / w) ** 0.5
    return x1 * w


@cython.nonecheck(False)
@cython.cdivision(True) 
cdef double random_levy(double c, double alpha): 
    cdef double u = M_PI * (random_uniform() - 0.5)
    cdef double v = 0.0
    cdef double t, s

    # Cauchy
    if alpha == 1.0:       
        t = tan(u)
        return c * t

    while v == 0:
        v = random_exponential(1.0)

    # Gaussian
    if alpha == 2.0:            
        t = 2 * sin(u) * sqrt(v)
        return c * t

    # General case
    t = sin(alpha * u) / pow(cos (u), 1 / alpha)
    s = pow(cos ((1 - alpha) * u) / v, (1 - alpha) / alpha)

    return c * t * s

@cython.boundscheck(False)
@cython.wraparound(False)
@cython.nonecheck(False)
@cython.cdivision(True)
cdef api double diffusion_trial(double v, double sv, double zr, double szr, double a, 
								double ndt, double sndt, double alpha, double dt, int max_steps):
    """
    INPUT:
    v         - drift rate (mean)
    sv        - variability in ndt
    zr        - relative starting point (bias) [0, 1]
    szr       - variability in relative starting point
    a         - threshold
    ndt       - non-decision time
    sndt      - variability in non-decision time
    alpha     - heavy-tailness of the noise distro
    dt        - time step (0.001 = 1 ms)
    max_steps - maximum number of steps before terminating trial simulation
    """

    # Declare variables
    cdef double n_steps = 0.0
    cdef double rt = 0.0
    cdef double rhs = pow(dt, 1. / alpha) # damping factor for noise
    # Include variabilities in parameters
    zr = zr - 0.5*szr + szr * random_uniform() # draw zr from block
    ndt = ndt - 0.5*sndt + sndt * random_uniform() # draw ndt from block
    v = v + sv * random_gaussian() # draw drift rate from Gaussian centered at v
    cdef double x = a * zr
    cdef double c = 1. / sqrt(2)

    # Simulate a single DM path
    while (x > 0 and x < a and n_steps < max_steps):

        # DDM equation
        x = x + v*dt + rhs*random_levy(c, alpha)
		
        # Increment step
        n_steps += 1.0

    rt = n_steps * dt 
    # Encode lower threshold with a negative sign
    rt = rt + ndt if x > 0 else -(rt + ndt)
    return rt