
from typing import Sequence

import keras

from bayesflow.experimental.simulation import Distribution, find_distribution
from bayesflow.experimental.types import Shape, Tensor
from .couplings import AllInOneCoupling


class CouplingFlow(keras.Sequential):
    """ Implements a coupling flow as a sequence of dual couplings with permutations and activation
    normalization. Incorporates ideas from [1-4].

    [1] Kingma, D. P., & Dhariwal, P. (2018).
    Glow: Generative flow with invertible 1x1 convolutions.
    Advances in Neural Information Processing Systems, 31.

    [2] Durkan, C., Bekasov, A., Murray, I., & Papamakarios, G. (2019).
    Neural spline flows. Advances in Neural Information Processing Systems, 32.

    [3] Ardizzone, L., Kruse, J., Lüth, C., Bracher, N., Rother, C., & Köthe, U. (2020).
    Conditional invertible neural networks for diverse image-to-image translation.
    In DAGM German Conference on Pattern Recognition (pp. 373-387). Springer, Cham.

    [4] Radev, S. T., Mertens, U. K., Voss, A., Ardizzone, L., & Köthe, U. (2020).
    BayesFlow: Learning complex stochastic models with invertible neural networks.
    IEEE Transactions on Neural Networks and Learning Systems.

    [5] Alexanderson, S., & Henter, G. E. (2020).
    Robust model training and generalisation with Studentising flows.
    arXiv preprint arXiv:2006.06599.
    """
    def __init__(self, couplings: Sequence[AllInOneCoupling], base_distribution: Distribution):
        super().__init__(couplings)
        self.base_distribution = base_distribution

    @classmethod
    def all_in_one(
            cls,
            target_dim: int,
            num_layers: 6,
            subnet_builder="default",
            transform="affine",
            permutation="fixed",
            act_norm=True,
            base_distribution="normal",
            **kwargs
    ) -> "CouplingFlow":
        """Construct a coupling flow, consisting of dual couplings with a single type of transform.

        Parameters
        ----------
        target_dim : int
            The dimensionality of the latent space, e.g., for estimating a model with 2 parameters, set
            ``target_dim=2``
        num_layers : int, optional, default: 6
            The number of dual coupling layers in the coupling flow. More layers will result in better
            performance for some applications at the cost of increased training time.
        subnet_builder : str or callable, optional, default: "default"
            Determines the structure of the internal networks used to generate the internal parameters
            for the coupling transforms. You can also pass a function that accepts a ``target_dim`` parameter
            and generates a custom architecture accordingly.

            The default builder will suffice for most applications. You can control the settings of the
            default networks by passing them as a dictionary into the ````subnet_settings```` optional keyword
            argument. For instance, to increase the dropout rate, you can do:
            subnet_settings=dict(dropout_rate=0.05).

            See below for a full list of settings.
        transform : str or callable, optional, default: "affine"
            The type of coupling transform used. Custom transforms can be passed as callable objects
            that implement a ``forward()`` and an ``inverse()`` method.

            Note: The string options are ``["affine", "spline"]``, where "spline" will typically result in
            better performance for low-dimensional problems at the cost of ~1.5x increase in training time.
        permutation : str, optional, default: "fixed"
            The type of permutation to apply between layers. Should be in ``["fixed", "learnable"]``
            Specifying a learnable permutation is advisable when you have many parameters and very few
            coupling layers to ensure proper mixing between dimensions (i.e., representation of correlations).
        act_norm : bool, optional, default: True
            A flag indicating whether to apply an invertible activation normalization layer prior to each
            coupling transformation. Don't touch unless you know what you are doing.
        base_distribution: str or callable, optional, default: "gaussian"
            The latent space distribution into which your targets are transformed. Currently implemented are:

            - "gaussian" : The standard choice, don't touch unless you know what you are doing.

            - "student": : Can help stabilize training by controlling the influence function of
                potentially problematic inputs in the training data, as suggested by [5].

            - "mixture"  : Can help with learning multimodal distribution, especially when using
                ``transform="affine"``, and you have some prior knowledge about the number of modes

            - callable   : Any other custom distribution implemented appropriately.
        **kwargs : dict, optional, default: {}

            Optional keyword arguments that will be passed to the ``subnet_builder`` or to the ``base_distribution``.

            For the ``subnet_builder``, you can pass a ``subnet_settings`` dictionary which can modify
            the following default settings:

            ``default_settings=dict(
                hidden_dim=512,
                num_hidden=2,
                activation="gelu",
                residual=True,
                spectral_norm=False,
                dropout_rate=0.05,
                zero_output_init=True
            )``

            For instance, to increase regularization for small data sets, you can pass:

             ``default_settings=dict(dropout_rate=0.2)``

            See the implementation of ``bayesflow.resnet.ConditionalResidualBlock`` for more details.

            For the ``base_distribution``  you can provide a ``base_distribution_parameters`` dictionary which is
            specific for each type of base distribution using.

            #TODO

        Returns
        -------
        flow : bayesflow.networks.CouplingFlow
            The callable coupling flow which be seamlessly interact with other keras objects.
        """

        base_distribution = find_distribution(base_distribution, shape=(target_dim,))

        couplings = []
        for _ in range(num_layers):
            layer = AllInOneCoupling(subnet_builder, target_dim, transform, permutation, act_norm, **kwargs)
            couplings.append(layer)

        return cls(couplings, base_distribution)

    def call(self, *args, **kwargs):
        return self.forward(*args, **kwargs)

    def compute_loss(self, x=None, y=None, y_pred=None, **kwargs):
        z, log_det = y_pred
        log_prob = self.base_distribution.log_prob(z)
        nll = -keras.ops.mean(log_prob + log_det)

        return nll

    def compute_metrics(self, x, y, y_pred, **kwargs):
        return {}

    def forward(self, targets, conditions=None, **kwargs) -> (Tensor, Tensor):
        latents = targets
        log_det = 0.
        for coupling in self.layers:
            latents, det = coupling.forward(latents, conditions, **kwargs)
            log_det += det

        return latents, log_det

    def inverse(self, latents, conditions=None) -> (Tensor, Tensor):
        targets = latents
        log_det = 0.
        for coupling in reversed(self.layers):
            targets, det = coupling.inverse(targets, conditions)
            log_det += det

        return targets, log_det

    def sample(self, batch_shape: Shape, conditions=None) -> Tensor:
        latents = self.base_distribution.sample(batch_shape)
        targets, _ = self.inverse(latents, conditions)

        return targets

    def log_prob(self, targets: Tensor, conditions=None, **kwargs) -> Tensor:
        latents, log_det = self.forward(targets, conditions, **kwargs)
        log_prob = self.base_distribution.log_prob(latents)

        return log_prob + log_det
