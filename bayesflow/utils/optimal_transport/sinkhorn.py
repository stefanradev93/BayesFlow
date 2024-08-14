import keras
import numpy as np

from bayesflow.types import Tensor

from ..dispatch import find_cost
from .. import logging
from ..numpy_utils import softmax


def sinkhorn(
    x1: Tensor,
    x2: Tensor,
    cost: str | Tensor = "euclidean",
    seed: int = None,
    numpy: bool = False,
    regularization: float = 1.0,
    max_steps: int = 10000,
    tolerance: float = 1e-6,
) -> (Tensor, Tensor):
    """
    Matches elements from x2 onto x1 using the Sinkhorn-Knopp algorithm.

    Sinkhorn-Knopp is an iterative algorithm that repeatedly normalizes the cost matrix into a doubly stochastic
    transport plan, containing assignment probabilities.
    The permutation is then sampled randomly according to the transport plan.

    :param x1: Tensor of shape (n, ...)
        Samples from the first distribution.

    :param x2: Tensor of shape (m, ...)
        Samples from the second distribution.

    :param cost: Method used to compute the transport cost. You may also pass the cost matrix directly.
        Default: 'euclidean'

    :param seed: Random seed used for the assignment. Required when using JAX backend.
        Default: None

    :param numpy: Whether to use numpy or keras backend.
        When using numpy, the computation is not differentiable, but ensured to remain on CPU.
         This may be necessary when this function is called in a multiprocessing context.

    :param regularization: Regularization parameter.
        Controls the standard deviation of the Gaussian kernel.
        Default: 1.0

    :param max_steps: Maximum number of iterations.
        Default: 1000

    :param tolerance: Absolute tolerance for convergence.
        Default: 1e-6

    :return: Tensors of shapes (n, ...) and (m, ...)
        x1 and x2 in optimal transport permutation order.
    """
    if numpy:
        return sinkhorn_numpy(
            x1=x1,
            x2=x2,
            cost=cost,
            seed=seed,
            regularization=regularization,
            max_steps=max_steps,
            tolerance=tolerance,
        )

    return sinkhorn_keras(
        x1=x1,
        x2=x2,
        cost=cost,
        seed=seed,
        regularization=regularization,
        max_steps=max_steps,
        tolerance=tolerance,
    )


def sinkhorn_keras(
    x1: Tensor, x2: Tensor, cost: str | Tensor, seed: int, regularization: float, max_steps: int, tolerance: float
) -> (Tensor, Tensor):
    cost = find_cost(cost, x1, x2, numpy=False)
    plan = sinkhorn_plan(
        cost=cost,
        regularization=regularization,
        max_steps=max_steps,
        tolerance=tolerance,
        numpy=False,
    )

    indices = keras.random.categorical(plan, num_samples=1, seed=seed)
    indices = keras.ops.squeeze(indices, axis=1)
    x2 = keras.ops.take(x2, indices, axis=0)

    return x1, x2


def sinkhorn_numpy(
    x1: np.ndarray,
    x2: np.ndarray,
    cost: str | Tensor,
    seed: int,
    regularization: float,
    max_steps: int,
    tolerance: float,
) -> (np.ndarray, np.ndarray):
    cost = find_cost(cost, x1, x2, numpy=True)
    plan = sinkhorn_plan(
        cost=cost,
        regularization=regularization,
        max_steps=max_steps,
        tolerance=tolerance,
        numpy=True,
    )

    if seed is not None:
        np.random.seed(seed)

    indices = []
    for row in range(len(x1)):
        index = np.random.choice(len(x2), p=plan[row])
        indices.append(index)

    x2 = np.take(x2, indices, axis=0)

    return x1, x2


def sinkhorn_plan(cost: Tensor, regularization: float, max_steps: int, tolerance: float, numpy: bool = False) -> Tensor:
    """
    Computes the Sinkhorn-Knopp optimal transport plan for the given cost matrix.

    :param cost: Tensor of shape (n, m).
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
    if numpy:
        return sinkhorn_plan_numpy(cost=cost, regularization=regularization, max_steps=max_steps, tolerance=tolerance)
    return sinkhorn_plan_keras(cost=cost, regularization=regularization, max_steps=max_steps, tolerance=tolerance)


def sinkhorn_plan_keras(cost: Tensor, regularization: float, max_steps: int, tolerance: float) -> Tensor:
    # scale regularization with the half-mean of the cost
    regularization = 0.5 * regularization * keras.ops.mean(cost)

    # initialize the transport plan from a gaussian kernel
    plan = keras.ops.exp(-0.5 * cost / regularization)

    def is_converged(plan):
        # check convergence: the plan should be doubly stochastic
        marginals = keras.ops.sum(plan, axis=0), keras.ops.sum(plan, axis=1)
        deviations = keras.ops.abs(marginals[0] - 1.0), keras.ops.abs(marginals[1] - 1.0)

        return keras.ops.logical_and(
            keras.ops.all(deviations[0] <= tolerance), keras.ops.all(deviations[1] <= tolerance)
        )

    def update_plan(plan):
        # Sinkhorn-Knopp: repeatedly normalize the transport plan along each dimension
        plan = keras.ops.softmax(plan, axis=0)
        plan = keras.ops.softmax(plan, axis=1)

        return plan

    plan = keras.ops.while_loop(lambda plan: ~is_converged(plan), update_plan, plan, maximum_iterations=max_steps)

    def do_nothing():
        pass

    def warn():
        marginals = keras.ops.sum(plan, axis=0)
        deviations = keras.ops.abs(marginals - 1.0)
        badness = 100.0 * keras.ops.max(deviations)

        msg = "Sinkhorn-Knopp did not converge after {:d} steps (badness: {:.1f}%)."

        logging.warning(msg, max_steps, badness)

    keras.ops.cond(is_converged(plan), do_nothing, warn)

    return plan


def sinkhorn_plan_numpy(cost: np.ndarray, regularization: float, max_steps: int, tolerance: float) -> np.ndarray:
    # scale regularization with the half-mean of the cost
    regularization = 0.5 * regularization * np.mean(cost)

    # initialize the transport plan from a gaussian kernel
    plan = np.exp(-0.5 * cost / regularization)

    for _ in range(max_steps):
        # check convergence: the plan should be doubly stochastic
        marginals = np.sum(plan, axis=0), np.sum(plan, axis=1)
        deviations = np.abs(marginals[0] - 1.0), np.abs(marginals[1] - 1.0)

        if np.all(deviations[0] < tolerance) and np.all(deviations[1] < tolerance):
            break

        # Sinkhorn-Knopp: repeatedly normalize the transport plan along each dimension
        plan = softmax(plan, axis=0)
        plan = softmax(plan, axis=1)

    marginals = np.sum(plan, axis=0)
    deviations = np.abs(marginals - 1.0)

    if np.any(deviations > tolerance):
        badness = 100.0 * np.max(deviations)
        msg = f"Sinkhorn-Knopp did not converge after {max_steps:d} steps (badness: {badness:.1f}%)."
        logging.warning(msg)

    return plan
