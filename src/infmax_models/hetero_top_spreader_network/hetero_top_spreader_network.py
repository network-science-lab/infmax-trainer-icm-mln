import torch
from torch.fx import Proxy
from torch.nn import ModuleList
from torch_geometric.nn import GraphNorm,  SAGEConv, Sequential
from torch_geometric.nn.conv import GATConv
from src.infmax_models.base.base import BaseHeteroModule


class HeteroTopSpreaderNetwork(BaseHeteroModule):
    def __init__(
        self,
        input_dim: int,
        hidden_channels: int,
        output_dim: int,
        heads: int | None = 4,
    ) -> None:
        super().__init__(is_hetero=False)
        self.layers = ModuleList()
        self.layers.append(
            Sequential(
                "x, edge_index",
                [
                    (
                        GATConv(
                            in_channels=input_dim,
                            out_channels=hidden_channels,
                            add_self_loops=False,
                            heads=heads,
                        ),
                        "x, edge_index -> x",
                    ),
                    (
                        GraphNorm(
                            in_channels=hidden_channels * heads,
                        ),
                        "x -> x",
                    ),
                    (torch.nn.Dropout(p=0.2), "x -> x"),
                    torch.nn.LeakyReLU(inplace=True),
                ],
            )
        )

        self.layers.append(
            Sequential(
                "x, edge_index",
                [
                    (
                        GATConv(
                            in_channels=hidden_channels * heads,
                            out_channels=hidden_channels,
                            add_self_loops=False,
                            heads=heads,
                        ),
                        "x, edge_index -> x",
                    ),
                    (
                        GraphNorm(
                            in_channels=hidden_channels * heads,
                        ),
                        "x -> x",
                    ),
                    (torch.nn.Dropout(p=0.2), "x -> x"),
                    torch.nn.LeakyReLU(inplace=True),
                ],
            )
        )
        
        self.head = SAGEConv(
            in_channels=hidden_channels * heads,
            out_channels=output_dim,
            aggr="mean",
        )

    def forward(
        self,
        x_dict: Proxy,
        z_dict: Proxy,
        edge_index_dict: Proxy,
    ) -> Proxy:
        for layer in self.layers:
            x_dict = layer(
                x=x_dict,
                edge_index=edge_index_dict,
            )

        x = self.head(
            x_dict,
            edge_index=edge_index_dict,
        )

        return x
