from typing import Iterable

from pandas import DataFrame
from sklearn.preprocessing import KBinsDiscretizer
from torch import Tensor, float32, long, tensor, zeros
from torch_geometric.data import HeteroData
from typing_extensions import Self

from src.utils.multilayer_network import MultilayerNetworkInfo


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

        data = HeteroData()
        data["actor"].x = zeros((len(network.actors_map), input_dim))
        data["actor"].y = cls._prepare_labels(
            output_dim=output_dim,
            network_info=network_info,
            df=df,
        )

        for idx, _ in enumerate(network.layers_order):
            layer_edge_indexes = network.adjacency_tensor[idx, ...].coalesce().indices()

            data["actor", f"{idx}_n", "actor"].edge_index = layer_edge_indexes

        return data

    @staticmethod
    def _prepare_labels(
        output_dim: int,
        network_info: MultilayerNetworkInfo,
        df: DataFrame,
    ) -> Tensor:
        match output_dim:
            case 0:
                raise AttributeError(
                    f"Output dimmension must be greater than 0: {output_dim}"
                )

            case 1:
                labels = tensor(
                    df[network_info.output_label_name].values.reshape(-1, 1),
                    dtype=float32,
                )

            case _:
                est = KBinsDiscretizer(
                    n_bins=output_dim,
                    encode="ordinal",
                    strategy="kmeans",
                )
                labels = est.fit_transform(
                    df[network_info.output_label_name].values.reshape(-1, 1)
                )
                labels = tensor(labels.squeeze(), dtype=long)

        return labels
