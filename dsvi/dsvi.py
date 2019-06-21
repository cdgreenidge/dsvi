"""An implementation of doubly-stochastic variational inference."""
import gpytorch.kernels as kernels
import torch
import torch.distributions as dist
import torch.nn as nn
import torch.quasirandom


class DGPLayer(nn.Module):
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

        # TODO: Add option for external initialization, of both correct and incorrect
        # dimensionality

        grid_1d = torch.linspace(-grid_bound, grid_bound, steps=grid_num)
        grid = grid_1d[:, None].expand(-1, input_dim)
        self.kernel = kernels.GridKernel(kernel, grid)

        self.register_buffer("input_dim", torch.tensor(input_dim))
        self.register_buffer("output_dim", torch.tensor(output_dim))

        self.inducing_locs = self.kernel.full_grid
        num_inducing = self.inducing_locs.size()[0]

        # Each of these parameters has shape (output_dim, num_inducing), i.e. they are
        # batched across output dimension
        self.inducing_means = nn.Parameter(
            torch.t(torch.quasirandom.SobolEngine(self.output_dim).draw(num_inducing))
        )
        self.inducing_scales = nn.Parameter(torch.ones(self.output_dim, num_inducing))

    def forward(self, x: torch.Tensor) -> torch.Tensor:  # type: ignore
        """Evaluate the GP on the inputs ``x``.

        Args:
            x: A tensor of shape ``(n, input_dim)``.

        Returns:
            In eval mode, the posterior mean of the GP predictions at ``x``. In train
            mode, a reparameterized sample from the posterior. Each is of shape
            ``(n, output_dim)``.

        """
        assert x.size()[1] == self.input_dim

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

        # f_cov no batch dim, so we will have to add one later.
        f_cov = self.kernel(x, x, diag=True) - (alpha * (kzz @ alpha)).sum(dim=0)

        # TODO: compute KL divergence regularization

        f_dist = dist.Independent(dist.Normal(f_mean, f_cov.unsqueeze(0)), 1)
        return f_dist.rsample().t()
