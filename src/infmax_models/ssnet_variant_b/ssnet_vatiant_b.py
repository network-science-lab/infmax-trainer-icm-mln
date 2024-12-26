import torch
from torch.nn import Dropout, LeakyReLU
from torch_geometric.nn import GCNConv, SAGEConv, Sequential

from _data_set.nsl_data_utils.loaders.constants import ACTOR
from src.infmax_models.base.base import BaseHeteroModule
from src.infmax_models.ssnet_variant_b import aggregator, spreader


class SSNetVariantB(BaseHeteroModule):
    """
    Super Spreaders Network Variant B.

    Idea backing this implementation is following:
    1. Compute embeddings of actors on each layer separately using the same trainable nn.modules
    2. Simulate spreading on each layer with custom self-attention layer
    3. Aggregate embeddings with trainable nn.module and predict SPs as `out_channels`-dim vector
    """

    def __init__(self, input_dim: int, hidden_channels: int, output_dim: int) -> None:
        """Initialise the object."""
        super().__init__(is_hetero=True)

        # use it to obtain embeddings of each relation as: tensor `nb_actors X nb_hidden_channels`
        self.layerwise_encoder = Sequential(
            "x_actors, x_edges",
            [
                (GCNConv(input_dim, hidden_channels // 8), "x_actors, x_edges -> x_interim"),
                LeakyReLU(inplace=True),
                (Dropout(p=0.2), "x_interim -> x_interim"),

                (SAGEConv(hidden_channels // 8, hidden_channels // 4, "mean"), "x_interim, x_edges -> x_interim"),
                LeakyReLU(inplace=True),
                (Dropout(0.2),"x_interim -> x_interim"),

                (SAGEConv(hidden_channels // 4, hidden_channels // 2, "mean"), "x_interim, x_edges -> x_interim"),
                LeakyReLU(inplace=True),
                (Dropout(0.2),"x_interim -> x_interim"),

                (SAGEConv(hidden_channels // 2, hidden_channels, "mean"), "x_interim, x_edges -> x_interim"),
                LeakyReLU(inplace=True),
                (Dropout(0.2),"x_interim -> x_interim"),
            ],
        )
        
        # simulate spreading in the hidden dimension
        self.spreading_surrogate_1 = spreader.CrossRelationAttention(hidden_channels, 2)
        self.spreading_surrogate_2 = spreader.CrossRelationAttention(hidden_channels // 2, 2)
        self.spreading_surrogate_3 = spreader.CrossRelationAttention(hidden_channels // 4, 2)
    
        # aggregate embeddings and obtain the final prediction
        self.layerwise_aggregator = aggregator.LayerwiseAggregation(hidden_channels // 8, output_dim)

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
        layers_emb = []
        z_mask = 1 - z_dict[ACTOR]

        # embed actors on each mln layer separately
        for layer_idx, layer_edges in enumerate(edge_index_dict.values()):
            layer_x = x_dict[ACTOR] * z_mask[:, layer_idx].unsqueeze(dim=1).expand_as(x_dict[ACTOR])
            layer_emb = self.layerwise_encoder(layer_x, layer_edges)
            layers_emb.append(layer_emb)

        # simulate multilayer spreading
        layers_emb = self.spreading_surrogate_1(layers_emb)
        layers_emb = self.spreading_surrogate_2(layers_emb)
        layers_emb = self.spreading_surrogate_3(layers_emb)

        # aggregate the embeddings and obtain final prediction
        layers_emb = torch.stack(layers_emb)
        y_pred = self.layerwise_aggregator(layers_emb)

        return {ACTOR: y_pred}
