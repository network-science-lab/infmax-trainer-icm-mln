"""A script where main data model is implemented."""

import logging

from typing import Iterable

import torch

from bidict import bidict
from sklearn.preprocessing import KBinsDiscretizer
from torch_geometric.data import HeteroData
from typing_extensions import Self

from _data_set.nsl_data_utils.loaders.centrality_loader import load_centralities
from _data_set.nsl_data_utils.loaders.constants import ACTOR, CENTRALITY_FUNCTIONS
from src.data_models.mln_info import MLNInfo


class MLNHeteroData(HeteroData):
    """
    Target class to store multilayergraph data for trained GNNs.

    Attributes:
    self["actor"].x - tensor of input features for actors of the network
    self["actor"].y - tensor of output labels denoting spreading potential for each actor
    self["actor"].z - tensor of mask with "1"s denoting nodes artificially added on each layer to 
        make the netwrork multiplex
    self["actor", "l_<idx>", "actor"] - tensor edges in layer named "l_<idx"
    self.network_name - name of the network
    self.actors_map - map of the actors' original names to their indices in tensors
    self.layers_map - map of the layers' original names to their indices in tensors
    """

    # TODO: In case of need use the following code as a starter for iter function
    def __iter__(self) -> Iterable:
        for key in self.stores:
            yield key

    @classmethod
    def from_network_info(
        cls, network_info: MLNInfo, output_dim: int, input_dim: int,
    ) -> Self:
        """Default constructor for this class."""
        data = cls()

        # tensor of input features
        data[ACTOR].x = cls._prepare_features(network_info=network_info, input_dim=input_dim)

        # tensor of edges; each layer denoted as another relation: [actor, l_<idx>, actor]
        for idx, _ in enumerate(network_info.mln_torch.layers_order):
            layer_edge_idx = network_info.mln_torch.adjacency_tensor[idx, ...].coalesce().indices()
            data[ACTOR, f"l_{idx}", ACTOR].edge_index = layer_edge_idx

        # tensor of labels to predict
        data[ACTOR].y = cls._prepare_labels(network_info=network_info, output_dim=output_dim)

        # mask of nodes that were added artificially to obtain multiplicity
        data[ACTOR].z = network_info.mln_torch.nodes_mask.T

        # auxiliary metadata to preserve original names of network, layers, and actors
        data.network_name = network_info.name
        data.actors_map = bidict(
            {
                str(actor): actors_map for
                actor, actors_map in network_info.mln_torch.actors_map.items()
            }
        )
        data.layers_map = bidict(
            {
                l_name: f"l_{l_idx}" for
                l_idx, l_name in enumerate(network_info.mln_torch.layers_order)
            }
        )

        return data

    @staticmethod
    def _prepare_features(network_info: MLNInfo, input_dim: int) -> torch.Tensor:
        logging.debug(f"Preparing features: {network_info.name}")

        if network_info.x_type == "zeros":
                return torch.zeros((len(network_info.mln_torch.actors_map), input_dim))
    
        elif  network_info.x_type == "centralities":
            if input_dim > len(CENTRALITY_FUNCTIONS) or input_dim <= 0:
                raise ValueError(
                    f"Input dim({input_dim}) must be > 0 and <= number of implemented centralities "
                    f"({len(CENTRALITY_FUNCTIONS)})"
                )
            features_df = load_centralities(
                network_name=network_info.mln_name,
                network_type=network_info.mln_type,
            )
            actors_order = [
                str(actor_id) for actor_id, actor_idx in sorted(
                    network_info.mln_torch.actors_map.items(), key=lambda id_idx: id_idx[1]
                )
            ]
            features_sorted_df = features_df.loc[actors_order]
            features_np = features_sorted_df.to_numpy()
            features_pt = torch.tensor(data=features_np, dtype=torch.float32)
            features_norm_pt = features_pt / len(network_info.mln.get_actors())
            return features_norm_pt[:, :input_dim]

        elif network_info.x_type == "scrapped":
            raise NotImplementedError(f"{network_info.x_type} has not been implemented yet")
        else:
            raise ValueError("Unknown x_type!")

    @staticmethod
    def _prepare_labels(network_info: MLNInfo, output_dim: int) -> torch.Tensor:
        logging.debug(f"Preparing labels: {network_info.name}")

        # convert the dataframe with raw data to have only values following the training setting
        sp_transformed = network_info.transform_sp()

        # this is for classification task  TODO: test it!
        if len(network_info.y_type) == 1:
            est = KBinsDiscretizer(n_bins=output_dim, encode="ordinal", strategy="kmeans")
            labels = est.fit_transform(sp_transformed[network_info.y_type].values.reshape(-1, 1))
            labels = torch.tensor(labels.squeeze(), dtype=torch.long)
        
        # this is for regression task
        else:
            Y_raw = sp_transformed[[ACTOR, *network_info.y_type]]
            Y_raw["actor_idx"] = Y_raw[ACTOR].map(network_info.mln_torch.actors_map)
            Y_raw = Y_raw.set_index("actor_idx").sort_index()
            labels = torch.tensor(Y_raw[network_info.y_type].values, dtype=torch.float32)
            labels = labels / len(sp_transformed[ACTOR])

        return labels
