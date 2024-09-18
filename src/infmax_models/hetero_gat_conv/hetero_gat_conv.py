from typing import Any

import torch.nn.functional as F
from torch import Tensor
from torch.fx import Proxy
from torch.nn import ModuleList
from torch_geometric.nn.conv import GATConv
from src.infmax_models.base.base import BaseHeteroModule

class GATHeteroGNN(BaseHeteroModule):
    def __init__(
        self,
        num_layers: int,
        input_dim: int,
        output_dim: int,
        params: list[dict[str, Any]],
    ) -> None:
        super().__init__(is_hetero=False)
        self.layers = ModuleList()
        self.layers.append(
            GATConv(
                in_channels=input_dim,
                **params[0],
            )
        )

        for i in range(1, num_layers - 1):
            in_channels = params[i - 1]["out_channels"] * params[i - 1]["heads"]
            self.layers.append(
                GATConv(
                    in_channels=in_channels,
                    **params[i],
                )
            )

        in_channels = params[-2]["out_channels"] * params[-2]["heads"]
        self.layers.append(
            GATConv(
                in_channels=in_channels,
                out_channels=output_dim,
                **params[-1],
            )
        )

    def forward(
        self,
        x_dict: Proxy,
        edge_index_dict: Proxy,
    ) -> Tensor:
        for layer in self.layers[:-1]:
            x_dict = F.relu(layer(x=x_dict, edge_index=edge_index_dict))

        x = self.layers[-1](x=x_dict, edge_index=edge_index_dict,)

        return x
