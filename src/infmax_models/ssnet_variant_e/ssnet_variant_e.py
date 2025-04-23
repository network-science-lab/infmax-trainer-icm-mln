from typing import Literal
import torch
from torch.nn import Linear
from torch_geometric.nn import GINConv, Sequential, GATConv, GCNConv

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

class SSNetVariantE(BaseHeteroModule):
    """
    Super Spreaders Network Variant D.

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
        num_layers: int = 3,
        layer_type: Literal[
            'GINConv',
            'GATConv',
            'GCNConv',
        ] = 'GINConv',
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

        self.input_dim = input_dim
        self.hidden_channels = hidden_channels
        self.output_dim = output_dim
        
        self.layerwise_encoder = Sequential(
            "x_actors, x_edges",
            [
                (
                    Sequential(
                        "x_actors, x_edges",
                        [
                            (
                                self.get_conv(
                                    in_channels = self.get_in_channel(i),
                                    out_channels = self.get_out_channel(i),
                                    layer_type = layer_type,
                                ),
                                "x_actors, x_edges -> x_actors",
                            ),
                            torch.nn.ReLU(inplace=True),
                        ],
                    ),
                    "x_actors, x_edges -> x_actors",
                )
                for i in range(num_layers)
            ],
        )

        if aggregation_type == LayerwiseAggregation.__name__:
            self.layerwise_aggregator = LayerwiseAggregation(self.get_out_channel(num_layers - 1))
        elif aggregation_type == MaxAggregation.__name__:
            self.layerwise_aggregator = MaxAggregation()
        elif aggregation_type == MinAggregation.__name__:
            self.layerwise_aggregator = MinAggregation()
        elif aggregation_type == AvgAggregation.__name__:
            self.layerwise_aggregator = AvgAggregation()
        elif aggregation_type == SumAggregation.__name__:
            self.layerwise_aggregator = SumAggregation()
        elif aggregation_type == AttentionAggregation.__name__:
            self.layerwise_aggregator = AttentionAggregation(self.get_out_channel(num_layers - 1))
        else:
            raise AttributeError("Incorrect name of the aggregator!")

        self.head = torch.nn.Sequential(
            torch.nn.Linear(self.get_out_channel(num_layers - 1), self.get_out_channel(num_layers - 1)),
            torch.nn.ReLU(inplace=True),
            torch.nn.Linear(self.get_out_channel(num_layers - 1), output_dim),
            torch.nn.Sigmoid(),
        )
        
    def get_in_channel(self,i:int) -> int:
        if i == 0:
            return self.input_dim
        if i == 1:
            return self.hidden_channels
        if i > 1:
            depth = (i - 1) * 2
            return self.hidden_channels // depth
        
    def get_out_channel(self,i:int) -> int:
        if i == 0:
            return self.hidden_channels
        if i > 0:
            depth = i * 2
            return self.hidden_channels // depth

    def get_conv(
        self,
        in_channels: int,
        out_channels: int,
        layer_type: Literal[
            'GINConv',
            'GATConv',
            'GCNConv',
        ] = 'GINConv',
    ) -> torch.nn.Module:
        match layer_type:
            case GINConv.__name__:
                return self.get_gin_layer(in_channels, out_channels)

            case GATConv.__name__:
                num_heads = 4
                return GATConv(in_channels, out_channels // num_heads, heads=num_heads)

            case GCNConv.__name__:
                return GCNConv(in_channels, out_channels)

            case _:
                raise AttributeError(f"Unknown convolution layer: {layer_type}")

    @staticmethod
    def get_gin_layer(in_channels: int, out_channels: int) -> torch.nn.Module:
        return GINConv(
            nn=torch.nn.Sequential(
                Linear(in_channels, out_channels),
                torch.nn.ReLU(inplace=True),
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
            y_relation = self.layerwise_encoder(layer_x, layer_edges)
            y_relations[layer_name] = y_relation

        # aggregate embeddings
        y_aggregated = self.layerwise_aggregator(y_relations)

        # obtain final prediction
        y_pred = self.head(y_aggregated)

        return {ACTOR: y_pred}
