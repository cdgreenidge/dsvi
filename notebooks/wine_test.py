"""Tests a Deep GP model on the breast cancer datset."""

#%%
from typing import Tuple

from gpytorch import kernels
import ignite
import ignite.engine
import sklearn.datasets
import sklearn.decomposition
import sklearn.model_selection
import sklearn.preprocessing
import torch
import torch.cuda
import torch.optim as optim
import torch.utils.data as data

import dsvi


def get_data_loaders() -> Tuple[data.DataLoader, data.DataLoader]:
    X_raw, y = sklearn.datasets.load_breast_cancer(return_X_y=True)
    X_scaled = sklearn.preprocessing.quantile_transform(
        X_raw, output_distribution="normal", n_quantiles=512, copy=True
    )

    #  It looks like we should reduce to around 6 or so. But to allow more inducing
    # points, we'll only keep the first 4
    X_reduced = sklearn.preprocessing.minmax_scale(
        sklearn.decomposition.PCA(1).fit_transform(X_scaled), feature_range=(-1, 1)
    )
    X_train, X_test, y_train, y_test = sklearn.model_selection.train_test_split(
        X_reduced, y, test_size=0.2
    )

    dtype = torch.float
    train_dataset = data.TensorDataset(
        torch.tensor(X_train, dtype=dtype), torch.tensor(y_train, dtype=dtype)
    )
    train_loader = data.DataLoader(
        train_dataset,
        batch_size=len(train_dataset),
        sampler=torch.utils.data.RandomSampler(
            train_dataset, replacement=True, num_samples=len(train_dataset)
        ),
        drop_last=True,
    )
    val_dataset = data.TensorDataset(
        torch.tensor(X_test, dtype=dtype), torch.tensor(y_test, dtype=dtype)
    )
    val_loader = data.DataLoader(
        val_dataset, batch_size=len(val_dataset), shuffle=False
    )
    return train_loader, val_loader


def run() -> None:
    train_loader, val_loader = get_data_loaders()
    num_data = len(train_loader) * train_loader.batch_size
    model = dsvi.DeepGP(
        (
            dsvi.Layer(
                kernels.ScaleKernel(kernels.RBFKernel()),
                input_dim=1,
                output_dim=1,
                grid_num=2,
            ),
        ),
        dsvi.LogisticBernoulli(),
    )

    device = "cpu"
    if torch.cuda.is_available():
        device = "cuda"

    optimizer = optim.Adam(model.parameters())

    def train_and_store_loss(
        engine: ignite.engine.Engine, batch: Tuple[torch.Tensor, torch.Tensor]
    ) -> float:
        inputs, targets = batch
        optimizer.zero_grad()
        loss = model.negative_elbo(
            inputs, targets.unsqueeze(-1), num_data=num_data, num_samples=1
        )
        loss.backward()
        optimizer.step()
        return loss.item()

    engine = ignite.engine.Engine(train_and_store_loss)
    engine.run(train_loader)


run()


#%%
