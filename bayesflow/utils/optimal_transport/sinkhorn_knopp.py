import keras
import logging

from bayesflow.types import Tensor
from bayesflow.utils import warning


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

    def is_converged(log_plan):
        # check convergence: the plan should be doubly stochastic
        # we only need to check axis 0 since we just normalized axis 1
        marginals = keras.ops.sum(keras.ops.exp(log_plan), axis=0)
        deviations = keras.ops.abs(marginals - 1.0)

        return keras.ops.all(deviations < tolerance)

    def update_plan(log_plan):
        # Sinkhorn-Knopp: repeatedly normalize the transport plan along each dimension
        log_plan = keras.ops.log_softmax(log_plan, axis=0)
        log_plan = keras.ops.log_softmax(log_plan, axis=1)

        return log_plan

    # scale regularization with the half-mean of the cost
    regularization = 0.5 * regularization * keras.ops.mean(cost_matrix)

    # initialize the transport plan from a logarithmic gaussian kernel
    log_plan = -0.5 * cost_matrix / regularization

    log_plan = keras.ops.while_loop(
        lambda log_plan: ~is_converged(log_plan), update_plan, log_plan, maximum_iterations=max_steps
    )

    def do_nothing():
        pass

    def warn():
        marginals = keras.ops.sum(keras.ops.exp(log_plan, axis=0))
        deviations = keras.ops.abs(marginals - 1.0)
        badness = 100.0 * keras.ops.max(deviations)
        badness = keras.ops.convert_to_tensor(badness, dtype="int32")

        logging.warning(f"Sinkhorn-Knopp did not converge after {max_steps} steps (badness: {badness}%).")

    keras.ops.cond(is_converged(log_plan), do_nothing, warn)

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

    def is_converged(plan):
        # check convergence: the plan should be doubly stochastic
        # we only need to check axis 0 since we just normalized axis 1
        marginals = keras.ops.sum(plan, axis=0)
        deviations = keras.ops.abs(marginals - 1.0)

        return keras.ops.all(deviations < tolerance)

    def update_plan(plan):
        # Sinkhorn-Knopp: repeatedly normalize the transport plan along each dimension
        plan = keras.ops.softmax(plan, axis=0)
        plan = keras.ops.softmax(plan, axis=1)

        return plan

    # scale regularization with the half-mean of the cost
    regularization = 0.5 * regularization * keras.ops.mean(cost_matrix)

    # initialize the transport plan from a gaussian kernel
    plan = keras.ops.exp(-0.5 * cost_matrix / regularization)

    plan = keras.ops.while_loop(lambda plan: ~is_converged(plan), update_plan, plan, maximum_iterations=max_steps)

    def do_nothing():
        pass

    def warn():
        marginals = keras.ops.sum(plan, axis=0)
        deviations = keras.ops.abs(marginals - 1.0)
        badness = 100.0 * keras.ops.max(deviations)

        msg = "Sinkhorn-Knopp did not converge after {:d} steps (badness: {:.1f}%)."

        warning(msg, max_steps, badness)

    keras.ops.cond(is_converged(plan), do_nothing, warn)

    return plan
