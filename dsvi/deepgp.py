"""An implementation of deep Gaussian Processes with DSVI inference."""
from typing import Iterable, Union

import gpytorch.kernels as kernels
import torch
import torch.distributions as dist
import torch.nn as nn
import torch.quasirandom


class DeepGPLayer(nn.Module):
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

        # TODO: Add option for external initialization, of both correct and incorrect
        # dimensionality

        grid_1d = torch.linspace(-grid_bound, grid_bound, steps=grid_num)
        grid = grid_1d[:, None].expand(-1, input_dim)
        self.kernel = kernels.GridKernel(kernel, grid)

        self.register_buffer("input_dim", torch.tensor(input_dim))
        self.register_buffer("output_dim", torch.tensor(output_dim))

        self.inducing_locs = self.kernel.full_grid
        self.register_buffer("num_inducing", torch.tensor(self.inducing_locs.size()[0]))

        # Each of these parameters has shape (output_dim, num_inducing), i.e. they are
        # batched across output dimension
        self.inducing_means = nn.Parameter(
            torch.quasirandom.SobolEngine(self.output_dim).draw(self.num_inducing).t()
        )
        self.inducing_scales = nn.Parameter(
            torch.ones(self.output_dim, self.num_inducing)
        )
        self.register_buffer("kl_regularization", torch.tensor(0.0))

    def forward(self, x: torch.Tensor) -> torch.Tensor:  # type: ignore
        """Evaluate the GP on the inputs ``x``.

        Args:
            x: A tensor of shape ``(n, input_dim)``.

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

        # Calculate linear transformation W for mean function
        if self.input_dim == 1:
            W = torch.eye(self.input_dim)
        else:
            W = torch.svd(x, some=False)[2][:, 0]  # PCA mapping

        kzz = self.kernel(self.inducing_locs, self.inducing_locs).add_jitter()
        kzx = self.kernel(self.inducing_locs, x).evaluate()
        alpha = kzz.inv_matmul(kzx)
        inducing_dist = dist.Independent(
            dist.Normal(self.inducing_means, self.inducing_scales), 1
        )

        # Mean function evaluated at x. Has size (1, num_x) so it broadcasts with
        # the batch dimensions of inducing_dist.
        m_x = (x @ W).unsqueeze(0)
        # Mean function evaluated at inducing locations. Has size (1, num_inducing).
        m_z = (self.inducing_locs @ W).unsqueeze(0)
        # The unsqueeze(-1) forces batch matrix multiplication across the output
        # dimensions, and the squeeze() removes the uneccessary dimension we added
        f_mean = (
            m_x + (alpha.t() @ ((inducing_dist.mean - m_z).unsqueeze(-1))).squeeze()
        )

        if not self.training:
            return f_mean.t()

        # f_cov has no batch dim, so we will have to add one later.
        f_cov = self.kernel(x, x, diag=True) - (alpha * (kzz @ alpha)).sum(dim=0)

        # We don't use PyTorch's KL divergence calculation because it doesn't take
        # advantage of GPyTorch
        inducing_cov = torch.diag_embed(inducing_dist.variance)
        trace = self.output_dim * torch.sum(
            torch.diagonal(kzz.inv_matmul(inducing_cov), dim1=-2, dim2=-1)
        )
        invquad, logdet_1 = kzz.inv_quad_logdet(self.inducing_means.t(), logdet=True)
        logdet_1 = self.output_dim * logdet_1  # Scale logdet 1 for output dimensions
        logdet_0 = torch.sum(torch.log(inducing_dist.variance))
        k = self.output_dim * self.num_inducing
        self.kl_regularization = 0.5 * (trace + invquad - k + logdet_1 - logdet_0)

        f_dist = dist.Independent(
            dist.Normal(f_mean, torch.sqrt(f_cov.unsqueeze(0))), 1
        )
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


class ExpPoisson(nn.Module):
    """A Poisson distribution output layer with exp nonlinearity."""

    def forward(self, x: torch.Tensor) -> dist.Independent:  # type: ignore
        """Output a Poisson distribution parameterized by ``exp(x)``."""
        _validate_input(x)
        return dist.Independent(dist.Poisson(torch.exp(x)), 1)


class Gaussian(nn.Module):
    """A Gaussian distribution output layer with trainable variance."""

    def __init__(self) -> None:
        super().__init__()
        self.scale = nn.Parameter(torch.tensor(1.0))

    def forward(self, x: torch.Tensor) -> dist.Independent:  # type: ignore
        """Output a Gaussian distribution parameterized by ``N(x, self.scale^2)``."""
        _validate_input(x)
        return dist.Independent(dist.Normal(x, self.scale), 1)


class LogisticBernoulli(nn.Module):
    """A Bernoulli distribution output layer with logistic nonlinearity."""

    def forward(self, x: torch.Tensor) -> dist.Independent:  # type: ignore
        """Output a Bernoulli distribution with probabilities ``sigmoid(x)``."""
        _validate_input(x)
        return dist.Independent(dist.Bernoulli(torch.sigmoid(x)), 1)


DeepGPLikelihood = Union[ExpPoisson, Gaussian, LogisticBernoulli]


class DeepGP(nn.Module):
    """A Deep Gaussian Process."""

    def __init__(
        self, layers: Iterable[DeepGPLayer], likelihood: DeepGPLikelihood
    ) -> None:
        pass
