"""Tests deepgp.py."""
import gpytorch.kernels as kernels
import pytest
import torch

from dsvi import deepgp


def test_dgplayer_can_compute_predictions():
    dgp = deepgp.Layer(kernels.RBFKernel(), input_dim=2, output_dim=3, grid_num=4)
    output = dgp(torch.rand(10, 2))
    assert output.size() == (10, 3)


def test_deepgplayer_computes_kl_regularization():
    dgp = deepgp.Layer(kernels.RBFKernel(), input_dim=2, output_dim=3, grid_num=4)
    dgp(torch.rand(10, 2), compute_kl=True)
    assert dgp.kl_regularization.item() > 0.0


def test_deepgplayer_can_turn_off_kl_regularization():
    dgp = deepgp.Layer(kernels.RBFKernel(), input_dim=2, output_dim=3, grid_num=4)
    dgp(torch.rand(10, 2), compute_kl=False)
    assert dgp.kl_regularization.item() == 0.0


def test_deepgplayer_checks_constructor_args():
    with pytest.raises(ValueError):
        deepgp.Layer(kernels.RBFKernel(), input_dim=0)
    with pytest.raises(ValueError):
        deepgp.Layer(kernels.RBFKernel(), output_dim=0)
    with pytest.raises(ValueError):
        deepgp.Layer(kernels.RBFKernel(), grid_bound=0)
    with pytest.raises(ValueError):
        deepgp.Layer(kernels.RBFKernel(), grid_num=0)


def test_deepgplayer_checks_input_dims():
    dgp = deepgp.Layer(kernels.RBFKernel(), input_dim=2, output_dim=3, grid_num=4)
    with pytest.raises(ValueError):
        dgp(torch.rand(10, 10))


likelihoods = (deepgp.ExpPoisson(), deepgp.Gaussian(), deepgp.LogisticBernoulli())


@pytest.mark.parametrize("likelihood", likelihoods)
def test_likelihoods_check_inputs(likelihood):
    with pytest.raises(ValueError):
        likelihood(torch.randn(5, 3))
    with pytest.raises(ValueError):
        likelihood(torch.randn(1, 1, 1))
    likelihood(torch.randn(10, 1))


def test_deepgp_can_compute_outputs():
    dgp = deepgp.DeepGP(
        layers=(deepgp.Layer(kernels.RBFKernel()), deepgp.Layer(kernels.RBFKernel())),
        likelihood=deepgp.ExpPoisson(),
    )
    dgp(torch.randn(10, 1))
    dgp.eval()
    dgp(torch.randn(10, 1))


def test_deepgp_can_compute_kl_regularization():
    dgp = deepgp.DeepGP(
        layers=(deepgp.Layer(kernels.RBFKernel()), deepgp.Layer(kernels.RBFKernel())),
        likelihood=deepgp.ExpPoisson(),
    )
    dgp(torch.randn(10, 1))
    assert torch.is_tensor(dgp.kl_regularization)
    assert dgp.kl_regularization > 0.0


def test_deepgp_can_compute_negative_elbo_loss():
    dgp = deepgp.DeepGP(
        layers=(deepgp.Layer(kernels.RBFKernel()), deepgp.Layer(kernels.RBFKernel())),
        likelihood=deepgp.ExpPoisson(),
    )
    elbo = dgp.negative_elbo(torch.randn(10, 1), torch.ones((10, 1)), num_data=10)
    assert torch.is_tensor(elbo)
