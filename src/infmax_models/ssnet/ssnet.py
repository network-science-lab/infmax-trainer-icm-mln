import torch
from torch.nn import Dropout
from torch_geometric.nn import GCNConv, SAGEConv, Sequential

from _data_set.nsl_data_utils.loaders.constants import ACTOR
from src.infmax_models.base.base import BaseHeteroModule
from src.infmax_models.ssnet.layerwise_aggregation import LayerwiseAggregation


class SSNet(BaseHeteroModule):
    """
    Super Spreaders Network.

    Idea backing this implementation is following:
    1. Compute embeddings of actors on each layer separately using the same trainable nn.modules
    2. Aggregate embeddings with trainable nn.module
    3. With Linear head predict spreading potentials as `out_channels`-dim vector
    """

    def __init__(
        self,
        input_dim: int,
        hidden_channels: int,
        output_dim: int,
    ) -> None:
        """Initialise the object."""
        super().__init__(is_hetero=True)
        self.layerwise_encoder = Sequential(
            "x_actors, x_edges",
            [
                # (Dropout(p=0.2), "x_actors -> x_actors"),  # TODO: verify if dropout also works with edges
                (
                    GCNConv(
                        in_channels=input_dim,
                        out_channels=hidden_channels // 2,
                        normalize=True,
                    ),
                    "x_actors, x_edges -> x_interim",
                ),
                torch.nn.LeakyReLU(inplace=True),
                (Dropout(p=0.2), "x_interim -> x_interim"),
                (
                    SAGEConv(
                        in_channels=hidden_channels // 2,
                        out_channels=hidden_channels,
                        aggr="mean",
                    ),
                    "x_interim, x_edges -> x_interim",
                ),
                torch.nn.LeakyReLU(inplace=True),
                (Dropout(p=0.2), "x_interim -> x_interim"),
                (
                    SAGEConv(
                        in_channels=hidden_channels,
                        out_channels=hidden_channels // 2,
                        aggr="mean",
                    ),
                    "x_interim, x_edges -> x_interim",
                ),
                torch.nn.LeakyReLU(inplace=True),
            ],
        )
        self.layerwise_aggregator = LayerwiseAggregation(hidden_channels // 2)
        self.head = torch.nn.Linear(
            in_features=hidden_channels // 2,
            out_features=output_dim,
        )

    def forward(
        self,
        x_dict: dict[str, torch.Tensor],
        edge_index_dict: dict[str, torch.Tensor],
    ) -> dict[str, torch.Tensor]:
        """
        Regress spreading potentials for actors of the given mln.

        :param data: multilayer network
        :return: spreading potentials as a vector of shape `[nb_mln_actors, out_channels]`
        """
        y_relations = []

        # embed actors on each mln layer separately
        for layer_edges in edge_index_dict.values():
            y_relation = self.layerwise_encoder(x_dict[ACTOR], layer_edges)  # TODO: we pass here centralities only. where is the mask for artificially added nodes?
            y_relations.append(y_relation)
        y_relations = torch.stack(y_relations)

        # aggregate embeddings
        y_aggregated = self.layerwise_aggregator(y_relations)

        # obtain final prediction
        y_pred = self.head(y_aggregated)  # TODO: enforce only positive values!!! errors can compenstate right now

        return {ACTOR: y_pred}
