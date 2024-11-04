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
                (
                    GCNConv(
                        in_channels=input_dim,
                        out_channels=hidden_channels // 4,
                        normalize=True,
                    ),
                    "x_actors, x_edges -> x_interim",
                ),
                torch.nn.LeakyReLU(inplace=True),
                (Dropout(p=0.2), "x_interim -> x_interim"),
                (
                    SAGEConv(
                        in_channels=hidden_channels // 4,
                        out_channels=hidden_channels // 2,
                        aggr="mean",
                    ),
                    "x_interim, x_edges -> x_interim",
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
                (Dropout(p=0.2), "x_interim -> x_interim"),
                (
                    SAGEConv(
                        in_channels=hidden_channels // 2,
                        out_channels=hidden_channels // 4,
                        aggr="mean",
                    ),
                    "x_interim, x_edges -> x_interim",
                ),
                torch.nn.LeakyReLU(inplace=True),
            ],
        )
        self.layerwise_aggregator = LayerwiseAggregation(hidden_channels // 4)
        self.head = torch.nn.Sequential(
            torch.nn.Linear(
                in_features=hidden_channels // 4,
                out_features=output_dim,
            ),
            torch.nn.Softplus(),  # ditto
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
        y_relations = []
        z_mask = 1 - z_dict[ACTOR]

        # embed actors on each mln layer separately
        for layer_idx, layer_edges in enumerate(edge_index_dict.values()):
            layer_x = x_dict[ACTOR] * z_mask[:, layer_idx].unsqueeze(dim=1).expand_as(x_dict[ACTOR])
            y_relation = self.layerwise_encoder(layer_x, layer_edges)
            y_relations.append(y_relation)
        y_relations = torch.stack(y_relations)

        # aggregate embeddings
        y_aggregated = self.layerwise_aggregator(y_relations)

        # obtain final prediction
        y_pred = self.head(y_aggregated)

        return {ACTOR: y_pred}
