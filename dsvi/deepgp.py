"""An implementation of deep Gaussian Processes with DSVI inference."""
from typing import Iterable, Union

import gpytorch.functions
import gpytorch.kernels as kernels
import gpytorch.utils.grid
import torch
import torch.distributions as dist
import torch.nn as nn
import torch.quasirandom


class Layer(nn.Module):
    """A deep Gaussian Process prior layer.

    Args:
        kernel: The GP kernel.
        input_dim: The dimension of the input.
        output_dim: The dimension of the output.
        grid_bound: The maximum/minimum value of the inducing point grid in each
            dimension, which is defined on a scaled unit hypercube.
        grid_num: The number of inducing points in each dimension.

    """

    def __init__(
        self,
        kernel: kernels.Kernel,
        input_dim: int = 1,
        output_dim: int = 1,
        grid_bound: float = 1.0,
        grid_num: int = 128,
    ):
        super().__init__()

        if input_dim <= 0:
            raise ValueError("Input dim must be positive, got {0}".format(input_dim))
        if output_dim <= 0:
            raise ValueError("Output dim must be positive, got {0}".format(output_dim))
        if grid_bound <= 0.0:
            raise ValueError("Grid bound must be positive, got {0}".format(grid_bound))
        if grid_num <= 0:
            raise ValueError("Grid num must be positive, got {0}".format(grid_num))

        self.kernel = kernel

        sobol = torch.quasirandom.SobolEngine(input_dim)
        inducing_locs = (2.0 * sobol.draw(grid_num) - 1.0) * grid_bound
        self.register_buffer("inducing_locs", inducing_locs)

        self.register_buffer("num_inducing", torch.tensor(self.inducing_locs.size()[0]))
        self.register_buffer("input_dim", torch.tensor(input_dim))
        self.register_buffer("output_dim", torch.tensor(output_dim))
        self.inducing_means = nn.Parameter(torch.zeros((output_dim, self.num_inducing)))
        self.inducing_scales = nn.Parameter(torch.eye(self.num_inducing))
        self.register_buffer("kl_regularization", torch.tensor(0.0))

    def forward(  # type: ignore
        self, x: torch.Tensor, compute_kl=True
    ) -> torch.Tensor:
        """Evaluate the GP on the inputs ``x``.

        Args:
            x: A tensor of shape ``(n, input_dim)``.
            compute_kl: Whether or not to compute the KL divergence regularization.

        Returns:
            In eval mode, the posterior mean of the GP predictions at ``x``. In train
            mode, a reparameterized sample from the posterior. Each is of shape
            ``(n, output_dim)``.

        """
        if x.dim() != 2 or x.size()[1] != self.input_dim:
            msg = "Input must be of size (n, {0}), got {1}".format(
                self.input_dim, tuple(x.size())
            )
            raise ValueError(msg)

        # Calculate linear transformation W for mean function, of size (output_dim, )
        if self.input_dim == 1:
            W = torch.ones((1,))
        else:
            W = torch.svd(x, some=False)[2][:, 0]  # PCA mapping

        kzz = self.kernel(self.inducing_locs, self.inducing_locs).add_jitter()
        kzx = self.kernel(self.inducing_locs, x).evaluate()
        alpha = gpytorch.functions.inv_matmul(kzz, kzx)
        inducing_dist = dist.MultivariateNormal(
            self.inducing_means, scale_tril=self.inducing_scales
        )

        mean_x = x @ W
        mean_z = self.inducing_locs @ W
        f_mean = mean_x + (
            alpha.t() @ (inducing_dist.mean - mean_z).unsqueeze(-1)
        ).squeeze(-1)
        if not self.training:
            return f_mean.t()

        S = inducing_dist.covariance_matrix
        kzz_eval = kzz.evaluate()
        f_cov = self.kernel(x, x, diag=True) - (alpha * ((kzz_eval - S) @ alpha)).sum(
            dim=1
        )

        if compute_kl:
            prior_dist = dist.MultivariateNormal(
                torch.zeros_like(self.inducing_means), kzz_eval
            )
            self.kl_regularization = dist.kl_divergence(inducing_dist, prior_dist).sum()

        f_dist = dist.Normal(f_mean, torch.sqrt(f_cov), validate_args=True)
        return f_dist.rsample().t()


def _validate_input(x: torch.Tensor) -> None:
    """Validate input for a likelihood (output) layer.

    Args:
        x: The input tensor to ``self.forward()``.

    Raises:
        ValueError: If X is not of shape ``(n, 1)``.

    """
    if x.dim() != 2 or x.size()[1] != 1:
        msg = "Input must be of size (n, 1), got {0}"
        raise ValueError(msg.format(tuple(x.size())))


class Likelihood(nn.Module):
    """A likelihood that takes a Tensor of parameters and outputs a distribution."""

    def __init__(self) -> None:
        super().__init__()

    def forward(self, x: torch.Tensor) -> dist.Distribution:  # type: ignore
        """Output a distribution parameterized by x."""


class ExpPoisson(Likelihood):
    """A Poisson distribution output layer with exp nonlinearity."""

    def forward(self, x: torch.Tensor) -> dist.Poisson:  # type: ignore
        """Output a Poisson distribution parameterized by ``exp(x)``."""
        _validate_input(x)
        return dist.Poisson(torch.exp(x))


class Gaussian(Likelihood):
    """A Gaussian distribution output layer with trainable variance."""

    def __init__(self) -> None:
        super().__init__()
        self.scale = nn.Parameter(torch.tensor(1.0))

    def forward(self, x: torch.Tensor) -> dist.Normal:  # type: ignore
        """Output a Gaussian distribution parameterized by ``N(x, self.scale^2)``."""
        _validate_input(x)
        return dist.Normal(x, self.scale)


class LogisticBernoulli(Likelihood):
    """A Bernoulli distribution output layer with logistic nonlinearity."""

    def forward(self, x: torch.Tensor) -> dist.Bernoulli:  # type: ignore
        """Output a Bernoulli distribution with probabilities ``sigmoid(x)``."""
        _validate_input(x)
        return dist.Bernoulli(torch.sigmoid(x))


DeepGPLikelihood = Union[ExpPoisson, Gaussian, LogisticBernoulli]


class DeepGP(nn.Module):
    """A Deep Gaussian Process.

    Args:
        layers: A list of Layer objects.
        likelihood: An output likelihood (one of ``ExpPoisson``, ``Gaussian``, or
            ``LogisticBernoulli``).

    """

    def __init__(self, layers: Iterable[Layer], likelihood: Likelihood) -> None:
        super().__init__()
        self.layers = nn.ModuleList(layers)
        self.likelihood = likelihood

    @property
    def kl_regularization(self) -> torch.Tensor:
        """Return the layer-wise sum KL regularization.

        This is usually computed during the ELBO evaluation.

        """
        kl_reg = sum(layer.kl_regularization for layer in self.layers)
        assert torch.is_tensor(kl_reg)
        return kl_reg  # type: ignore

    def forward(  # type: ignore
        self, x: torch.Tensor, compute_kl=True
    ) -> dist.Distribution:
        """Evaluate the DeepGP model at the points in ``x``.

        Args:
            x: A tensor of shape ``(n, input_dim_1)``, where input_dim_1 is the input
            dimension of the first layer.
            compute_kl: A boolean controlling whether or not to compute the KL
                divergence.

        Returns:
            A distribution batched over ``n`` representing the Deep GP output.

        """
        for layer in self.layers:
            x = layer(x, compute_kl)
        return self.likelihood(x)

    def negative_elbo(
        self, x: torch.Tensor, y: torch.Tensor, num_data: int, num_samples: int = 8
    ) -> torch.Tensor:
        """Compute the negative ELBO loss.

        Args:
            x: a batch of inputs, of size ``(n, input_dim)``, where ``input_dim`` is the
                input dimension of the first layer in the DeepGP.
            y: a batch of labels, of size ``(n, output_dim)``, where ``output_dim`` is
                the output dimension of the last layer in the DeepGP.
            num_data: The number of items in the dataset.

        Returns:
            A scalar Tensor, the negative ELBO loss.

        """
        n = x.size()[0]
        input_dim = self.layers[0].input_dim.item()
        output_dim = self.layers[-1].output_dim.item()
        if tuple(x.size()) != (n, input_dim):
            msg = "Expected x of size {0}, got {1}"
            raise ValueError(msg.format((n, input_dim), tuple(x.size())))
        if tuple(y.size()) != (n, output_dim):
            msg = "Expected y of size {0}, got {1}"
            raise ValueError(msg.format((n, output_dim), tuple(y.size())))

        samples = []
        for _ in range(num_samples - 1):
            samples.append(self.forward(x, compute_kl=False).log_prob(y).sum())
        samples.append(self.forward(x, compute_kl=True).log_prob(y).sum())
        likelihood = sum(samples) / num_samples
        scaling = num_data / x.size()[0]

        elbo = scaling * likelihood - self.kl_regularization
        return -elbo
