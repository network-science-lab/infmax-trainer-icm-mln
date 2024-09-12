from typing import Callable

import numpy as np
from sklearn.preprocessing import KBinsDiscretizer
from torch import tensor, zeros
from torch_geometric.data import Dataset, HeteroData
from torch_geometric.typing import EdgeType, NodeType

from src.hetero_data.hetero_data import LightningHeteroData
from src.utils.multilayer_network import MultilayerNetworkInfo


class DataFrameHeteroDataset(Dataset):
    def __init__(
        self,
        networks: list[MultilayerNetworkInfo],
        input_dim: int,
        output_dim: int,
        root: str,
        transform: Callable | None = None,
        pre_transform: Callable | None = None,
    ) -> None:
        super().__init__(
            root=root,
            transform=transform,
            pre_transform=pre_transform,
        )

        self._input_dim = input_dim
        self._output_dim = output_dim

        self.data_list = [
            self._network_to_hetero_data(
                network_info=network_info,
            )
            for network_info in networks
        ]

    def _network_to_hetero_data(
        self, network_info: MultilayerNetworkInfo
    ) -> LightningHeteroData:
        network = network_info.network
        df = (
            network_info.output_df[["actor", network_info.label_name]]
            .groupby(["actor"])
            .mean()
            .reset_index()
        )

        est = KBinsDiscretizer(
            n_bins=self._output_dim,
            encode="ordinal",
            strategy="kmeans",
        )
        labels = est.fit_transform(np.array(df[network_info.label_name]).reshape(-1, 1))
        labels = tensor(labels.reshape(1, -1))[0].long()

        data = LightningHeteroData()
        data[network_info.network_name].x = zeros(
            (len(network.actors_map), self._input_dim)
        )
        data[network_info.network_name].y = labels

        for idx, layer in enumerate(network.layers_order):
            layer_edge_indexes = network.adjacency_tensor.indices()[
                :, network.adjacency_tensor.indices()[0] == idx
            ][1:]
            # Recommended for hetero data to use onlu letters, numbers and '_'
            layer = layer.replace("-", "_")

            data[
                network_info.network_name, layer, network_info.network_name
            ].edge_index = layer_edge_indexes

        return data

    def len(self) -> int:
        return len(self.data_list)

    def get(self, idx) -> HeteroData:
        return self.data_list[idx]

    def get_metadata(self) -> tuple[list[NodeType], list[EdgeType]]:
        nodes_data = []
        edges_data = []
        for data in self.data_list:
            node_data, edge_data = data.metadata()
            nodes_data.extend(node_data)
            edges_data.extend(edge_data)

        return nodes_data, edges_data
