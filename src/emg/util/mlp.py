from typing import Type

import torch.nn as nn


def mlp(
    dims: list[int],
    activation: Type[nn.Module],
    last_activation: Type[nn.Module] | None = None,
    dropout: float | None = None,
    norm: Type[nn.Module] | None = None,
) -> nn.Sequential:
    layers = []
    for i in range(len(dims) - 1):
        layers.append(nn.Linear(dims[i], dims[i + 1]))
        if i < len(dims) - 2:
            if norm is not None:
                layers.append(norm(dims[i + 1]))
            layers.append(activation())
            if dropout is not None:
                layers.append(nn.Dropout(dropout))
        elif last_activation is not None:
            layers.append(last_activation())
    return nn.Sequential(*layers)
