from typing import Literal
import torch
from torch_geometric.utils import dropout
from torch.nn import BatchNorm1d, Linear, Dropout
from torch_geometric.nn import GINConv, Sequential

from _data_set.nsl_data_utils.loaders.constants import ACTOR
from src.infmax_models.base.base import BaseHeteroModule
from src.infmax_models.ssnet.aggregation import (
    MaxAggregation,
    MinAggregation,
    AvgAggregation,
    SumAggregation,
    LayerwiseAggregation,
    AttentionAggregation,
)

class SSNetVariantC(BaseHeteroModule):
    """
    Super Spreaders Network Variant C.

    Idea backing this implementation is following:
    1. Compute embeddings of actors on each layer separately using the same trainable nn.modules
    2. Aggregate embeddings
    3. With Linear head predict spreading potentials as `out_channels`-dim vector
    """

    def __init__(
        self,
        input_dim: int,
        hidden_channels: int,
        output_dim: int,
        aggregation_type: Literal[
            "MaxAggregation",
            "MinAggregation",
            "AvgAggregation",
            "SumAggregation",
            "LayerwiseAggregation",
            "AttentionAggregation",
        ] = "LayerwiseAggregation"
    ) -> None:
        """Initialise the object."""
        super().__init__(is_hetero=True)

        self.layerwise_encoder = Sequential(
            "x_actors, x_edges",
            [
                (
                    self.get_gin_layer(input_dim, hidden_channels // 4),
                    "x_actors, x_edges -> x_interim",
                ),
                (BatchNorm1d(hidden_channels // 4), "x_interim -> x_interim"),
                torch.nn.LeakyReLU(inplace=True),
                (Dropout(p=0.2), "x_interim -> x_interim"),
                (
                    self.get_gin_layer(hidden_channels // 4, hidden_channels // 2),
                    "x_interim, x_edges -> x_interim",
                ),
                (BatchNorm1d(hidden_channels // 2), "x_interim -> x_interim"),
                torch.nn.LeakyReLU(inplace=True),
                (Dropout(p=0.2), "x_interim -> x_interim"),
                (
                    self.get_gin_layer(hidden_channels // 2, hidden_channels),
                    "x_interim, x_edges -> x_interim",
                ),
                (BatchNorm1d(hidden_channels), "x_interim -> x_interim"),
                torch.nn.LeakyReLU(inplace=True),
                (Dropout(p=0.2), "x_interim -> x_interim"),
                (
                    self.get_gin_layer(hidden_channels, hidden_channels),
                    "x_interim, x_edges -> x_interim",
                ),
                (BatchNorm1d(hidden_channels), "x_interim -> x_interim"),
                torch.nn.LeakyReLU(inplace=True),
                (Dropout(p=0.2), "x_interim -> x_interim"),
            ],
        )

        if aggregation_type == LayerwiseAggregation.__name__:
            self.layerwise_aggregator = LayerwiseAggregation(hidden_channels)
        elif aggregation_type == MaxAggregation.__name__:
            self.layerwise_aggregator = MaxAggregation()
        elif aggregation_type == MinAggregation.__name__:
            self.layerwise_aggregator = MinAggregation()
        elif aggregation_type == AvgAggregation.__name__:
            self.layerwise_aggregator = AvgAggregation()
        elif aggregation_type == SumAggregation.__name__:
            self.layerwise_aggregator = SumAggregation()
        elif aggregation_type == AttentionAggregation.__name__:
            self.layerwise_aggregator = AttentionAggregation(hidden_channels)
        else:
            raise AttributeError("Incorrect name of the aggregator!")

        # self.edge_dropout = DropoutEdge(p=0.1)
        self.head = torch.nn.Sequential(
            torch.nn.Linear(hidden_channels, hidden_channels // 2),
            torch.nn.LeakyReLU(inplace=True),
            torch.nn.Linear(hidden_channels // 2, output_dim),
            torch.nn.Sigmoid(),
        )
    
    @staticmethod
    def get_gin_layer(in_channels: int, out_channels: int) -> torch.nn.Module:
        return GINConv(
            nn=torch.nn.Sequential(
                Linear(in_channels, out_channels),
                torch.nn.LeakyReLU(inplace=True),
                Linear(out_channels, out_channels),
            ),
            train_eps=True,
        )

    def forward(
        self,
        x_dict: dict[str, torch.Tensor],
        z_dict: dict[str, torch.Tensor],
        edge_index_dict: dict[str, torch.Tensor],
    ) -> dict[str, torch.Tensor]:
        """
        Regress spreading potentials for actors of the given mln.

        :param x_dict: dict with actors' features `{"actor": torch.Tensor}`
        :param z_dict: dict with the mask of artificially added nodes `{"actor": torch.Tensor}`
        :param edge_index_dict: dict with edges `{("actor", "<layer_idx>", "actor"): torch.Tensor}`

        :return: spreading potentials as a vector of shape `[nb_mln_actors, out_channels]`
        """
        y_relations = {}
        z_mask = 1 - z_dict[ACTOR]

        # embed actors on each mln layer separately
        for layer_idx, (layer_name, layer_edges) in enumerate(edge_index_dict.items()):
            layer_x = x_dict[ACTOR] * z_mask[:, layer_idx].unsqueeze(dim=1).expand_as(
                x_dict[ACTOR]
            )
            # edges = self.edge_dropout(layer_edges)
            y_relation = self.layerwise_encoder(layer_x, layer_edges)
            y_relations[layer_name] = y_relation

        # aggregate embeddings
        y_aggregated = self.layerwise_aggregator(y_relations)

        # obtain final prediction
        y_pred = self.head(y_aggregated)

        return {ACTOR: y_pred}


class DropoutEdge(torch.nn.Module):
    def __init__(
        self,
        p: float = 0.2,
        relabel_nodes: bool = False,
    ) -> None:
        super().__init__()
        self._p = p
        self._force_undirected = relabel_nodes
        
    def forward(self, edge_index: torch.Tensor) -> torch.Tensor:
        edges, edge_mask = dropout.dropout_edge(
            edge_index=edge_index,
            p=self._p,
            force_undirected=self._force_undirected,
            training=self.training,
        )
        
        return edges
