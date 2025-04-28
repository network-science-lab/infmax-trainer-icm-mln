"""A script where main data model is implemented."""

import logging
from typing import Iterable

import network_diffusion as nd
import pandas as pd
import torch
from bidict import bidict
from torch_geometric.data import HeteroData
from typing_extensions import Self

from _data_set.nsl_data_utils.loaders.centrality_loader import load_centralities
from _data_set.nsl_data_utils.loaders.constants import ACTOR, CENTRALITY_FUNCTIONS
from _data_set.nsl_data_utils.loaders.net_loader import load_network
from _data_set.nsl_data_utils.loaders.sp_loader import load_sp
from src.data_models.mln_info import MLNInfo
import networkx as nx


class MLNHeteroData(HeteroData):
    """
    Target class to store multilayer graph data for training.

    Attributes:
    self["actor"].x - tensor of input features for actors of the network
    self["actor"].y - tensor of output labels denoting spreading potential for each actor
    self["actor"].z - tensor of mask with "1"s denoting nodes artificially added on each layer to
        make the netwrork multiplex
    self["actor", "l_<idx>", "actor"] - tensor edges in layer named "l_<idx"
    self.network_type - type of the network
    self.network_name - name of the network
    self.actors_map - map of the actors' original names to their indices in tensors
    self.layers_map - map of the layers' original names to their indices in tensors
    self.y_names - list[str] with names of labels that are stored in `self.y` attribute
    """

    def __iter__(self) -> Iterable:
        for key in self.stores:
            yield key

    @classmethod
    def from_network_info(
        cls,
        network_info: MLNInfo,
        input_dim: int,
        output_dim: int,
    ) -> Self:
        # read the network itself
        mln = load_network(
            network_info.mln_type,
            network_info.mln_name,
            as_tensor=False,
        )

        diameter = 0 # TODO: CLEAN DIRTY CODE
        for layer in mln.layers:
            try:
                diameter += nx.diameter(mln[layer])
            except:
                largest_cc = max(nx.connected_components(mln[layer]), key=len)
                g_largest_cc = mln[layer].subgraph(largest_cc).copy()
                diameter += nx.diameter(g_largest_cc)
        diameter = diameter / len(mln.layers)
        del mln

        mln_torch = load_network(
            network_info.mln_type,
            network_info.mln_name,
            as_tensor=True,
        )

        return cls.from_mln_network(
            mln_torch=mln_torch,
            network_info=network_info,
            input_dim=input_dim,
            output_dim=output_dim,
            diameter=diameter,
        )

    @classmethod
    def from_mln_network(
        cls,
        mln_torch: nd.MultilayerNetworkTorch,
        network_info: MLNInfo,
        input_dim: int,
        output_dim: int,
        diameter: int,
        sp_df: pd.DataFrame | None = None,
    ) -> Self:
        """Default constructor for this class."""
        data = cls()

        # tensor of input features
        data.diameter = diameter
        data[ACTOR].x = cls._prepare_features(network_info, mln_torch, input_dim)

        # tensor of edges; each layer denoted as another relation: [actor, l_<idx>, actor]
        for idx, _ in enumerate(mln_torch.layers_order):
            layer_edge_idx = mln_torch.adjacency_tensor[idx, ...].coalesce().indices()
            data[ACTOR, f"l_{idx}", ACTOR].edge_index = layer_edge_idx

        # tensor of labels to predict
        data[ACTOR].y = cls._prepare_labels(network_info, mln_torch, sp_df)
        data.y_names = network_info.y_type

        # mask of nodes that were added artificially to obtain multiplicity
        data[ACTOR].z = mln_torch.nodes_mask.T

        # auxiliary metadata to preserve original names of network, layers, and actors
        data.network_type = network_info.mln_type
        data.network_name = network_info.mln_name
        data.actors_map = bidict(
            {
                str(actor): actors_map
                for actor, actors_map in mln_torch.actors_map.items()
            }
        )
        data.layers_map = bidict(
            {
                l_name: f"l_{l_idx}"
                for l_idx, l_name in enumerate(mln_torch.layers_order)
            }
        )

        return data

    @staticmethod
    def _prepare_features(
        network_info: MLNInfo,
        network_torch: nd.MultilayerNetworkTorch,
        input_dim: int,
    ) -> torch.Tensor:
        logging.debug(f"Preparing features: {network_info.name}")

        if network_info.x_type == "zeros":
            return torch.zeros((len(network_torch.actors_map), input_dim))
    
        elif network_info.x_type == "ones":
            return torch.ones((len(network_torch.actors_map), input_dim))

        elif network_info.x_type == "centralities":
            if input_dim > len(CENTRALITY_FUNCTIONS) or input_dim <= 0:
                raise ValueError(
                    f"Input dim({input_dim}) must be > 0 and <= number of implemented centralities "
                    f"({len(CENTRALITY_FUNCTIONS)})"
                )
            features_df = load_centralities(network_info.ft_path)
            actors_order = [
                str(actor_id)
                for actor_id, actor_idx in sorted(
                    network_torch.actors_map.items(), key=lambda id_idx: id_idx[1]
                )
            ]
            features_sorted_df = features_df.loc[actors_order]
            features_np = features_sorted_df.to_numpy()
            features_pt = torch.tensor(data=features_np, dtype=torch.float32)
            return features_pt[:, :input_dim]

        elif network_info.x_type == "scrapped":
            raise NotImplementedError(f"{network_info.x_type} has not been implemented yet")

        else:
            raise ValueError("Unknown x_type!")

    @staticmethod
    def _prepare_labels(
        network_info: MLNInfo,
        network_torch: nd.MultilayerNetworkTorch,
        sp_df: pd.DataFrame | None = None,
    ) -> torch.Tensor:
        logging.debug(f"Preparing labels: {network_info.name}")

        # obtain a dataframe from valid paths following the training setting
        if sp_df is None:
            sp_df = load_sp(network_info.sp_paths)

        # aggregate values over actors
        Y_raw = sp_df[[ACTOR, *network_info.y_type]]
        Y_raw = Y_raw.groupby(ACTOR).mean().reset_index()
        Y_raw["actor_idx"] = Y_raw[ACTOR].map(network_torch.actors_map)
        Y_raw = Y_raw.set_index("actor_idx").sort_index()

        labels = torch.tensor(Y_raw[network_info.y_type].values, dtype=torch.float32)
        return labels
