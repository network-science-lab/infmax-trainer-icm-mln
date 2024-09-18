from typing import Iterable

from torch_geometric.data import HeteroData
from src.utils.multilayer_network import MultilayerNetworkInfo
from torch import tensor, zeros, long
from sklearn.preprocessing import KBinsDiscretizer
from typing_extensions import Self
import numpy as np

class LightningHeteroData(HeteroData):
    def __iter__(self) -> Iterable:
        for key in self.stores:
            yield key

    @classmethod
    def from_network_info(
        cls, 
        network_info: MultilayerNetworkInfo, 
        output_dim: int, 
        input_dim: int,
    ) -> Self:
        network = network_info.network
        df = (
            network_info.spreading_potential[["actor", network_info.output_label_name]]
            .groupby(["actor"])
            .mean()
            .reset_index()
        )

        est = KBinsDiscretizer(
            n_bins=output_dim,
            encode="ordinal",
            strategy="kmeans",
        )
        labels = est.fit_transform(df[network_info.output_label_name].values.reshape(-1, 1))
        labels = tensor(labels.squeeze(), dtype=long)

        data = HeteroData()
        data["actor"].x = zeros(
            (len(network.actors_map), input_dim)
        )
        data["actor"].y = labels

        for idx, _ in enumerate(network.layers_order):
            layer_edge_indexes = network.adjacency_tensor[idx, ...].coalesce().indices()

            data[
                "actor", f"{idx}_n", "actor"
            ].edge_index = layer_edge_indexes

        return data
    