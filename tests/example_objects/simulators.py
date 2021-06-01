import numpy as np
from numba import njit


@njit
def diffusion_trial(v, a, ndt, zr, dt, max_steps):
    """Simulates a trial from the diffusion model."""

    n_steps = 0.
    x = a * zr

    # Simulate a single DM path
    while (x > 0 and x < a and n_steps < max_steps):
        # DDM equation
        x += v * dt + np.sqrt(dt) * np.random.normal()

        # Increment step
        n_steps += 1.0

    rt = n_steps * dt
    return rt + ndt if x > 0. else -rt - ndt


@njit
def diffusion_condition(n_trials, v, a, ndt, zr=0.5, dt=0.005, max_steps=1e4):
    """Simulates a diffusion process over an entire condition."""

    x = np.empty(n_trials)
    for i in range(n_trials):
        x[i] = diffusion_trial(v, a, ndt, zr, dt, max_steps)
    return x


@njit
def diffusion_2_conds(params, n_trials, dt=0.005, max_steps=1e4):
    """
    Simulates a diffusion process for 2 conditions with 5 parameters (v1, v2, a1, a2, ndt).
    """

    n_trials_c1 = n_trials[0]
    n_trials_c2 = n_trials[1]

    v1, v2, a1, a2, ndt = params
    rt_c1 = diffusion_condition(n_trials_c1, v1, a1, ndt, dt=dt, max_steps=max_steps)
    rt_c2 = diffusion_condition(n_trials_c2, v2, a2, ndt, dt=dt, max_steps=max_steps)
    rts = np.concatenate((rt_c1, rt_c2))
    return rts


def dm_batch_simulator(prior_samples, n_obs, dt=0.005, s=1.0, max_iter=1e4):
    """
    Simulate multiple diffusion_model_datasets.
    """

    n_sim = prior_samples.shape[0]
    sim_data = np.empty((n_sim, n_obs), dtype=np.float32)

    n1 = n_obs // 2
    n2 = n_obs - n1

    # Simulate diffusion data
    for i in range(n_sim):
        sim_data[i] = diffusion_2_conds(prior_samples[i], (n1, n2))

    # Create condition labels
    cond_arr = np.stack(n_sim * [np.concatenate((np.zeros(n1), np.ones(n2)))])
    sim_data = np.stack((sim_data, cond_arr), axis=-1)

    return sim_data


@njit
def forward_model1(params, n_obs, V0=-70, I_input=3, dt=0.2):
    # HH-2pars

    # pars = [gbar_Na, gbar_K]
    # I_input = input current in muA/cm2
    # I_duration = duration of current input in ms
    # dt = dt

    I_duration = n_obs
    gbar_Na, gbar_K = params

    # fixed parameters
    tau_max = 6e2  # ms
    Vt = -60.  # mV
    nois_fact = 0.1  # uA/cm2
    E_leak = -70.  # mV
    E_Na = 53  # mV
    E_K = -107  # mV
    C = 1
    g_l = 0.1
    gbar_M = 0.07

    tstep = float(dt)

    ####################################
    # Current (I) muA/cm2
    t_on = 10
    t_post = 10
    I_duration = np.round(n_obs * dt - t_on - t_post - dt, 2)
    assert I_duration > 0, "Please provide n_obs >= 106!"
    t_off = I_duration + t_post

    t = np.arange(0, np.round(t_on + t_off + dt, 2), dt)

    I = np.zeros_like(t)
    I[int(np.round(t_on / dt)):int(np.round(t_off / dt))] = I_input

    ####################################
    # kinetics
    def efun(z):
        if np.abs(z) < 1e-4:
            return 1 - z / 2
        else:
            return z / (np.exp(z) - 1)

    def alpha_m(x):
        v1 = x - Vt - 13.
        return 0.32 * efun(-0.25 * v1) / 0.25

    def beta_m(x):
        v1 = x - Vt - 40
        return 0.28 * efun(0.2 * v1) / 0.2

    def alpha_h(x):
        v1 = x - Vt - 17.
        return 0.128 * np.exp(-v1 / 18.)

    def beta_h(x):
        v1 = x - Vt - 40.
        return 4.0 / (1 + np.exp(-0.2 * v1))

    def alpha_n(x):
        v1 = x - Vt - 15.
        return 0.032 * efun(-0.2 * v1) / 0.2

    def beta_n(x):
        v1 = x - Vt - 10.
        return 0.5 * np.exp(-v1 / 40)

    # steady-states and time constants
    def tau_n(x):
        return 1 / (alpha_n(x) + beta_n(x))

    def n_inf(x):
        return alpha_n(x) / (alpha_n(x) + beta_n(x))

    def tau_m(x):
        return 1 / (alpha_m(x) + beta_m(x))

    def m_inf(x):
        return alpha_m(x) / (alpha_m(x) + beta_m(x))

    def tau_h(x):
        return 1 / (alpha_h(x) + beta_h(x))

    def h_inf(x):
        return alpha_h(x) / (alpha_h(x) + beta_h(x))

    # slow non-inactivating K+
    def p_inf(x):
        v1 = x + 35.
        return 1.0 / (1. + np.exp(-0.1 * v1))

    def tau_p(x):
        v1 = x + 35.
        return tau_max / (3.3 * np.exp(0.05 * v1) + np.exp(-0.05 * v1))

    ####################################
    # simulation from initial point
    V = np.zeros_like(t)  # voltage
    n = np.zeros_like(t)
    m = np.zeros_like(t)
    h = np.zeros_like(t)
    p = np.zeros_like(t)

    V[0] = float(V0)
    n[0] = n_inf(V[0])
    m[0] = m_inf(V[0])
    h[0] = h_inf(V[0])
    p[0] = p_inf(V[0])

    for i in range(1, t.shape[0]):
        tau_V_inv = ((m[i - 1] ** 3) * gbar_Na * h[i - 1] + (n[i - 1] ** 4) * gbar_K + g_l + gbar_M * p[i - 1]) / C
        V_inf = ((m[i - 1] ** 3) * gbar_Na * h[i - 1] * E_Na + (n[i - 1] ** 4) * gbar_K * E_K + g_l * E_leak + gbar_M *
                 p[i - 1] * E_K
                 + I[i - 1] + nois_fact * np.random.randn() / (tstep ** 0.5)) / (tau_V_inv * C)
        V[i] = V_inf + (V[i - 1] - V_inf) * np.exp(-tstep * tau_V_inv)
        n[i] = n_inf(V[i]) + (n[i - 1] - n_inf(V[i])) * np.exp(-tstep / tau_n(V[i]))
        m[i] = m_inf(V[i]) + (m[i - 1] - m_inf(V[i])) * np.exp(-tstep / tau_m(V[i]))
        h[i] = h_inf(V[i]) + (h[i - 1] - h_inf(V[i])) * np.exp(-tstep / tau_h(V[i]))
        p[i] = p_inf(V[i]) + (p[i - 1] - p_inf(V[i])) * np.exp(-tstep / tau_p(V[i]))

    return np.expand_dims(V, -1)


@njit
def forward_model2(params, n_obs, V0=-70, I_input=3, dt=0.2):
    # HH-3pars

    # pars = [gbar_Na, gbar_K, gbar_M]
    # I_input = input current in muA/cm2
    # I_duration = duration of current input in ms
    # dt = dt

    I_duration = n_obs
    gbar_Na, gbar_K, gbar_M = params

    # fixed parameters
    tau_max = 6e2  # ms
    Vt = -60.  # mV
    nois_fact = 0.1  # uA/cm2
    E_leak = -70.  # mV
    E_Na = 53  # mV
    E_K = -107  # mV
    C = 1
    g_l = 0.1

    tstep = float(dt)

    ####################################
    # Current (I) muA/cm2
    t_on = 10
    t_post = 10
    I_duration = np.round(n_obs * dt - t_on - t_post - dt, 2)
    assert I_duration > 0, "Please provide n_obs >= 106!"
    t_off = I_duration + t_post

    t = np.arange(0, np.round(t_on + t_off + dt, 2), dt)

    I = np.zeros_like(t)
    I[int(np.round(t_on / dt)):int(np.round(t_off / dt))] = I_input

    ####################################
    # kinetics
    def efun(z):
        if np.abs(z) < 1e-4:
            return 1 - z / 2
        else:
            return z / (np.exp(z) - 1)

    def alpha_m(x):
        v1 = x - Vt - 13.
        return 0.32 * efun(-0.25 * v1) / 0.25

    def beta_m(x):
        v1 = x - Vt - 40
        return 0.28 * efun(0.2 * v1) / 0.2

    def alpha_h(x):
        v1 = x - Vt - 17.
        return 0.128 * np.exp(-v1 / 18.)

    def beta_h(x):
        v1 = x - Vt - 40.
        return 4.0 / (1 + np.exp(-0.2 * v1))

    def alpha_n(x):
        v1 = x - Vt - 15.
        return 0.032 * efun(-0.2 * v1) / 0.2

    def beta_n(x):
        v1 = x - Vt - 10.
        return 0.5 * np.exp(-v1 / 40)

    # steady-states and time constants
    def tau_n(x):
        return 1 / (alpha_n(x) + beta_n(x))

    def n_inf(x):
        return alpha_n(x) / (alpha_n(x) + beta_n(x))

    def tau_m(x):
        return 1 / (alpha_m(x) + beta_m(x))

    def m_inf(x):
        return alpha_m(x) / (alpha_m(x) + beta_m(x))

    def tau_h(x):
        return 1 / (alpha_h(x) + beta_h(x))

    def h_inf(x):
        return alpha_h(x) / (alpha_h(x) + beta_h(x))

    # slow non-inactivating K+
    def p_inf(x):
        v1 = x + 35.
        return 1.0 / (1. + np.exp(-0.1 * v1))

    def tau_p(x):
        v1 = x + 35.
        return tau_max / (3.3 * np.exp(0.05 * v1) + np.exp(-0.05 * v1))

    ####################################
    # simulation from initial point
    V = np.zeros_like(t)  # voltage
    n = np.zeros_like(t)
    m = np.zeros_like(t)
    h = np.zeros_like(t)
    p = np.zeros_like(t)

    V[0] = float(V0)
    n[0] = n_inf(V[0])
    m[0] = m_inf(V[0])
    h[0] = h_inf(V[0])
    p[0] = p_inf(V[0])

    for i in range(1, t.shape[0]):
        tau_V_inv = ((m[i - 1] ** 3) * gbar_Na * h[i - 1] + (n[i - 1] ** 4) * gbar_K + g_l + gbar_M * p[i - 1]) / C
        V_inf = ((m[i - 1] ** 3) * gbar_Na * h[i - 1] * E_Na + (n[i - 1] ** 4) * gbar_K * E_K + g_l * E_leak + gbar_M *
                 p[i - 1] * E_K
                 + I[i - 1] + nois_fact * np.random.randn() / (tstep ** 0.5)) / (tau_V_inv * C)
        V[i] = V_inf + (V[i - 1] - V_inf) * np.exp(-tstep * tau_V_inv)
        n[i] = n_inf(V[i]) + (n[i - 1] - n_inf(V[i])) * np.exp(-tstep / tau_n(V[i]))
        m[i] = m_inf(V[i]) + (m[i - 1] - m_inf(V[i])) * np.exp(-tstep / tau_m(V[i]))
        h[i] = h_inf(V[i]) + (h[i - 1] - h_inf(V[i])) * np.exp(-tstep / tau_h(V[i]))
        p[i] = p_inf(V[i]) + (p[i - 1] - p_inf(V[i])) * np.exp(-tstep / tau_p(V[i]))

    return np.expand_dims(V, -1)


@njit
def forward_model3(params, n_obs, V0=-70, I_input=3, dt=0.2):
    # HH-4pars

    # pars = [gbar_l, gbar_Na, gbar_K, gbar_M]
    # I_input = input current in muA/cm2
    # I_duration = duration of current input in ms
    # dt = dt

    I_duration = n_obs
    g_l, gbar_Na, gbar_K, gbar_M = params

    # fixed parameters
    tau_max = 6e2  # ms
    Vt = -60.  # mV
    nois_fact = 0.1  # uA/cm2
    E_leak = -70.  # mV
    E_Na = 53  # mV
    E_K = -107  # mV
    C = 1

    tstep = float(dt)

    ####################################
    # Current (I) muA/cm2
    t_on = 10
    t_post = 10
    I_duration = np.round(n_obs * dt - t_on - t_post - dt, 2)
    assert I_duration > 0, "Please provide n_obs >= 106!"
    t_off = I_duration + t_post

    t = np.arange(0, np.round(t_on + t_off + dt, 2), dt)

    I = np.zeros_like(t)
    I[int(np.round(t_on / dt)):int(np.round(t_off / dt))] = I_input

    ####################################
    # kinetics
    def efun(z):
        if np.abs(z) < 1e-4:
            return 1 - z / 2
        else:
            return z / (np.exp(z) - 1)

    def alpha_m(x):
        v1 = x - Vt - 13.
        return 0.32 * efun(-0.25 * v1) / 0.25

    def beta_m(x):
        v1 = x - Vt - 40
        return 0.28 * efun(0.2 * v1) / 0.2

    def alpha_h(x):
        v1 = x - Vt - 17.
        return 0.128 * np.exp(-v1 / 18.)

    def beta_h(x):
        v1 = x - Vt - 40.
        return 4.0 / (1 + np.exp(-0.2 * v1))

    def alpha_n(x):
        v1 = x - Vt - 15.
        return 0.032 * efun(-0.2 * v1) / 0.2

    def beta_n(x):
        v1 = x - Vt - 10.
        return 0.5 * np.exp(-v1 / 40)

    # steady-states and time constants
    def tau_n(x):
        return 1 / (alpha_n(x) + beta_n(x))

    def n_inf(x):
        return alpha_n(x) / (alpha_n(x) + beta_n(x))

    def tau_m(x):
        return 1 / (alpha_m(x) + beta_m(x))

    def m_inf(x):
        return alpha_m(x) / (alpha_m(x) + beta_m(x))

    def tau_h(x):
        return 1 / (alpha_h(x) + beta_h(x))

    def h_inf(x):
        return alpha_h(x) / (alpha_h(x) + beta_h(x))

    # slow non-inactivating K+
    def p_inf(x):
        v1 = x + 35.
        return 1.0 / (1. + np.exp(-0.1 * v1))

    def tau_p(x):
        v1 = x + 35.
        return tau_max / (3.3 * np.exp(0.05 * v1) + np.exp(-0.05 * v1))

    ####################################
    # simulation from initial point
    V = np.zeros_like(t)  # voltage
    n = np.zeros_like(t)
    m = np.zeros_like(t)
    h = np.zeros_like(t)
    p = np.zeros_like(t)

    V[0] = float(V0)
    n[0] = n_inf(V[0])
    m[0] = m_inf(V[0])
    h[0] = h_inf(V[0])
    p[0] = p_inf(V[0])

    for i in range(1, t.shape[0]):
        tau_V_inv = ((m[i - 1] ** 3) * gbar_Na * h[i - 1] + (n[i - 1] ** 4) * gbar_K + g_l + gbar_M * p[i - 1]) / C
        V_inf = ((m[i - 1] ** 3) * gbar_Na * h[i - 1] * E_Na + (n[i - 1] ** 4) * gbar_K * E_K + g_l * E_leak + gbar_M *
                 p[i - 1] * E_K
                 + I[i - 1] + nois_fact * np.random.randn() / (tstep ** 0.5)) / (tau_V_inv * C)
        V[i] = V_inf + (V[i - 1] - V_inf) * np.exp(-tstep * tau_V_inv)
        n[i] = n_inf(V[i]) + (n[i - 1] - n_inf(V[i])) * np.exp(-tstep / tau_n(V[i]))
        m[i] = m_inf(V[i]) + (m[i - 1] - m_inf(V[i])) * np.exp(-tstep / tau_m(V[i]))
        h[i] = h_inf(V[i]) + (h[i - 1] - h_inf(V[i])) * np.exp(-tstep / tau_h(V[i]))
        p[i] = p_inf(V[i]) + (p[i - 1] - p_inf(V[i])) * np.exp(-tstep / tau_p(V[i]))

    return np.expand_dims(V, -1)
