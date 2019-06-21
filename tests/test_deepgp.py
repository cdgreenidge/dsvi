"""Tests deepgp.py."""
import gpytorch.kernels as kernels
import pytest
import torch

from dsvi import deepgp


def test_dgplayer_can_compute_predictions():
    dgp = deepgp.DGPLayer(kernels.RBFKernel(), input_dim=2, output_dim=3, grid_num=4)
    output = dgp(torch.rand(10, 2))
    assert output.size() == (10, 3)


def test_dgplayer_checks_constructor_args():
    with pytest.raises(ValueError):
        deepgp.DGPLayer(kernels.RBFKernel(), input_dim=0)
    with pytest.raises(ValueError):
        deepgp.DGPLayer(kernels.RBFKernel(), output_dim=0)
    with pytest.raises(ValueError):
        deepgp.DGPLayer(kernels.RBFKernel(), grid_bound=0)
    with pytest.raises(ValueError):
        deepgp.DGPLayer(kernels.RBFKernel(), grid_num=0)


def test_dgplayer_checks_input_dims():
    dgp = deepgp.DGPLayer(kernels.RBFKernel(), input_dim=2, output_dim=3, grid_num=4)
    with pytest.raises(ValueError):
        dgp(torch.rand(10, 10))
