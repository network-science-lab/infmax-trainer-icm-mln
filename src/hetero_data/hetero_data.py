from typing import Iterable

from pandas import DataFrame
from sklearn.preprocessing import KBinsDiscretizer
from torch import Tensor, float32, long, tensor, zeros
from torch_geometric.data import HeteroData
from typing_extensions import Self

from _data_set.nsl_data_utils.loaders.constants import ACTOR, PROTOCOL
from src.utils.multilayer_network import MultilayerNetworkInfo


class LightningHeteroData(HeteroData):
    # TODO: In case of need use the following code as a starter for iter function
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
        spreading_potential_columns = [ACTOR, PROTOCOL]
        if isinstance(network_info.output_label_name, list):
            spreading_potential_columns.extend(network_info.output_label_name)
        else:
            spreading_potential_columns.append(network_info.output_label_name)

        df = network_info.spreading_potential[spreading_potential_columns]
        df = df[df[PROTOCOL] == network_info.protocol].drop(PROTOCOL, axis=1)
        df = df.groupby([ACTOR]).mean().reset_index()

        data = HeteroData()
        # TODO: ADD different feature generation
        data[ACTOR].x = zeros((len(network.actors_map), input_dim))
        data[ACTOR].y = cls._prepare_labels(
            output_dim=output_dim,
            network_info=network_info,
            df=df,
        )

        for idx, _ in enumerate(network.layers_order):
            layer_edge_indexes = network.adjacency_tensor[idx, ...].coalesce().indices()

            data[ACTOR, f"n_{idx}", ACTOR].edge_index = layer_edge_indexes

        return data

    @staticmethod
    def _prepare_labels(
        output_dim: int,
        network_info: MultilayerNetworkInfo,
        df: DataFrame,
    ) -> Tensor:
        match network_info.output_label_name:
            case None:
                raise AttributeError(
                    f"Output dimmension must be greater than 0: {output_dim}"
                )

            case list():
                Y_raw = df[network_info.output_label_name]
                Y_raw["actor_idx"] = Y_raw.index.map(network_info.network.actors_map)
                Y_raw = Y_raw.set_index("actor_idx").sort_index()
                labels = tensor(
                    Y_raw.values,
                    dtype=float32,
                )

            case str():
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
