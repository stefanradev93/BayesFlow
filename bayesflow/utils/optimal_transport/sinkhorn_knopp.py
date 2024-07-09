import keras
import logging

from bayesflow.types import Tensor


def sinkhorn_log(
    cost_matrix: Tensor, regularization: float = 1.0, max_steps: int = 1000, tolerance: float = 1e-6
) -> Tensor:
    """
    Computes the Sinkhorn-Knopp optimal transport plan for the given cost matrix.
    This algorithm is stabilized by performing the computations in logarithmic space.

    :param cost_matrix: Tensor of shape (n, m).
        Defines the transport costs between samples.
    :param regularization: Regularization parameter.
        Controls the standard deviation of the Gaussian kernel.
        Default: 1.0
    :param max_steps: Maximum number of iterations.
        Default: 1000
    :param tolerance: Absolute tolerance for convergence.
        Default: 1e-6
    :return: Tensor of shape (n, m)
        The logarithmic transport probabilities.
    """
    # scale regularization with the half-mean of the cost
    regularization = 0.5 * regularization * keras.ops.mean(cost_matrix)

    # initialize the transport plan from a logarithmic gaussian kernel
    log_plan = -0.5 * cost_matrix / regularization

    marginal = float("inf")
    for _ in range(max_steps):
        # Sinkhorn-Knopp: repeatedly normalize the transport plan along each dimension
        log_plan = keras.ops.log_softmax(log_plan, axis=0)
        log_plan = keras.ops.log_softmax(log_plan, axis=1)

        # check convergence: the plan should be doubly stochastic
        # we only need to check axis 0 since we just normalized axis 1
        marginal = keras.ops.sum(keras.ops.exp(log_plan), axis=0)
        is_converged = keras.ops.all(keras.ops.abs(marginal - 1.0) < tolerance)

        if is_converged:
            return log_plan

    badness = keras.ops.max(keras.ops.abs(marginal - 1.0))
    logging.warning(f"Sinkhorn-Knopp did not converge after {max_steps} steps (badness: {badness:.1e}).")

    return log_plan


def sinkhorn_knopp(
    cost_matrix: Tensor, regularization: float = 1.0, max_steps: int = 1000, tolerance: float = 1e-6
) -> Tensor:
    """
    Computes the Sinkhorn-Knopp optimal transport plan for the given cost matrix.

    :param cost_matrix: Tensor of shape (n, m).
        Defines the transport costs between samples.
    :param regularization: Regularization parameter.
        Controls the standard deviation of the Gaussian kernel.
        Default: 1.0
    :param max_steps: Maximum number of iterations.
        Default: 1000
    :param tolerance: Absolute tolerance for convergence.
        Default: 1e-6
    :return: Tensor of shape (n, m)
        The transport probabilities.
    """
    # scale regularization with the half-mean of the cost
    regularization = 0.5 * regularization * keras.ops.mean(cost_matrix)

    # initialize the transport plan from a logarithmic gaussian kernel
    plan = keras.ops.exp(-0.5 * cost_matrix / regularization)

    marginal = float("inf")
    for _ in range(max_steps):
        # Sinkhorn-Knopp: repeatedly normalize the transport plan along each dimension
        plan = keras.ops.softmax(plan, axis=0)
        plan = keras.ops.softmax(plan, axis=1)

        # check convergence: the plan should be doubly stochastic
        # we only need to check axis 0 since we just normalized axis 1
        marginal = keras.ops.sum(plan, axis=0)
        is_converged = keras.ops.all(keras.ops.abs(marginal - 1.0) < tolerance)

        if is_converged:
            return plan

    badness = keras.ops.max(keras.ops.abs(marginal - 1.0))
    logging.warning(f"Sinkhorn-Knopp did not converge after {max_steps} steps (badness: {badness:.1e}).")

    return plan
