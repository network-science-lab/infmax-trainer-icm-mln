import logging
from typing import Iterable

import bidict
import numpy as np
import torch
from network_diffusion.mln import MLNetworkActor
from network_diffusion.mln.functions import remove_selfloop_edges
from pandas import DataFrame
from sklearn.preprocessing import KBinsDiscretizer
from torch_geometric.data import HeteroData
from typing_extensions import Self

from _data_set.nsl_data_utils.loaders.constants import ACTOR, PROTOCOL
from src.hetero_data import CENTRALITY_FUNCTIONS
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

        data = cls()
        data[ACTOR].x = cls._prepare_features(
            network_info=network_info,
            input_dim=input_dim,
        )
        data[ACTOR].y = cls._prepare_labels(
            output_dim=output_dim,
            network_info=network_info,
            df=df,
        )

        data.network_name = network_info.network_name
        data.actors_map = bidict.bidict(
            {str(actor): actors_map for actor, actors_map in network.actors_map.items()}
        )
        data.layers_map = bidict.bidict(
            {l_name: f"l_{l_idx}" for l_idx, l_name in enumerate(network.layers_order)}
        )
        data[ACTOR].z = network.nodes_mask.T

        for idx, _ in enumerate(network.layers_order):
            layer_edge_indexes = network.adjacency_tensor[idx, ...].coalesce().indices()

            data[ACTOR, f"l_{idx}", ACTOR].edge_index = layer_edge_indexes

        return data

    @staticmethod
    def _prepare_features(
        network_info: MultilayerNetworkInfo,
        input_dim: int,
    ) -> torch.Tensor:
        logging.info(f"Preparing features: {network_info.network_name}")
        network_info.network = remove_selfloop_edges(network_info.network)
        match network_info.features_type:
            case None:
                raise ValueError(
                    f"Feature name must be passed and dimension must be greater than 0: {input_dim}"
                )
            case "zeros":
                return torch.zeros((len(network_info.network.actors_map), input_dim))
            case "centralities":
                if input_dim > len(CENTRALITY_FUNCTIONS) or input_dim <= 0:
                    raise ValueError(
                        f"Input dim({input_dim}) must be greater than 0 and "
                        "lower or equal number of implemented "
                        f"centralities({len(CENTRALITY_FUNCTIONS)})"
                    )

                mln_centralities: list[dict[MLNetworkActor, float]] = [
                    centrality_function(network_info.mln_network)
                    for centrality_function in CENTRALITY_FUNCTIONS[:input_dim]
                ]

                features_raw = []
                actor_indices = []
                for actor in network_info.mln_network.get_actors():
                    actor_indices.append(
                        network_info.network.actors_map[actor.actor_id]
                    )
                    features_raw.append(
                        [
                            mln_centrality[actor] if actor in mln_centrality else 0
                            for mln_centrality in mln_centralities
                        ]
                    )
                values = np.array(features_raw)

                actor_indices = np.array(actor_indices)
                sorted_features = values[actor_indices.argsort()]

                features = torch.tensor(
                    data=sorted_features,
                    dtype=torch.float32,
                )
                features = features / len(features_raw)

                return features

            case "scraped":
                # TODO
                raise NotImplementedError(
                    f"{network_info.features_type} has not been implemented yet"
                )

    @staticmethod
    def _prepare_labels(
        output_dim: int,
        network_info: MultilayerNetworkInfo,
        df: DataFrame,
    ) -> torch.Tensor:
        logging.info(f"Preparing labels: {network_info.network_name}")
        if not network_info.output_label_name:
            raise AttributeError(
                (
                    "Output label name must be passed and "
                    f"dimmension must be greater than 0: {output_dim}"
                )
            )

        if isinstance(network_info.output_label_name, str):
            est = KBinsDiscretizer(
                n_bins=output_dim,
                encode="ordinal",
                strategy="kmeans",
            )
            labels = est.fit_transform(
                df[network_info.output_label_name].values.reshape(-1, 1)
            )
            labels = torch.tensor(labels.squeeze(), dtype=torch.long)
        elif isinstance(network_info.output_label_name, list):
            Y_raw = df[network_info.output_label_name]
            Y_raw["actor_idx"] = Y_raw.index.map(network_info.network.actors_map)
            Y_raw = Y_raw.set_index("actor_idx").sort_index()
            labels = torch.tensor(
                Y_raw.values,
                dtype=torch.float32,
            )
            labels = labels / len(df[ACTOR])

        return labels
