"""Loader the network of fMRI scans of 74 healthy women in resting-state."""

from pathlib import Path
from tqdm import tqdm

import networkx as nx
import network_diffusion as nd
import pandas as pd


def _parse_adj_mats(network_dir: str, binary: bool, thresh: float | None) -> dict[str, nx.Graph]:
    """Convert directory of adjacency matrix files into dictionary of edgelists."""
    layers = {}
    for network_file in tqdm(Path(network_dir).glob("*.csv")):
        try:
            # read as pandas DataFrame, index=source, col=target
            layer = pd.read_csv(network_file, index_col=0)
            if layer.shape[0] != layer.shape[1]:
                raise ValueError("Expecting matrix with index as source and column as target!")
            if thresh is not None:
                layer[layer <= thresh] = 0
            if binary:
                layer[layer != 0] = 1
            # ensure that index (node name) is string, since word2vec will need it as str
            if pd.api.types.is_numeric_dtype(layer.index):
                layer.index = layer.index.map(str)
            # convert matrix --> adjacency list
            layer = layer.stack(future_stack=True).reset_index()
            # rename columns
            layer.columns = ["source", "target", "weight"]
            # remove null weights
            layer = layer[~(layer["weight"] == 0.0)].reset_index(drop=True)
            layers[network_file.name] = layer
        except Exception as e:
            print(f"Could not read file '{network_file}': {e}")
    return layers


def _parse_edgelists(edgelists: dict[str, nx.Graph]) -> nd.MultilayerNetwork:
    """Convert dict of edgelists to dict of nx.Graphs."""
    l_names, l_graphs = [], []
    for l_name, l_edge_list in edgelists.items():
        l_names.append(l_name)
        l_graphs.append(nx.convert_matrix.from_pandas_edgelist(l_edge_list, edge_attr="weight"))
    return l_names, l_graphs


def read_fmri74(network_dir: str, binary: bool, thresh: float | None) -> nd.MultilayerNetwork:
    """
    Read the network of fMRI scans of 74 healthy women in resting-state.

    The dataset was posted to the 1000 Functional Connectomes Project and used in work "Analysis of
    Population Functional Connectivity Data via Multilayer Network Embeddings" at Network Science,
    DOI: 10.1017/nws.2020.39

    :param network_dir: directory of adjacency matrix files
    :param binary: boolean of whether or not to convert edge weights to binar
    :param thresh: threshold for edge weights; it will accepts weights <= thresh

    :return: multilayer network read from given directory.
    """
    edge_lists = _parse_adj_mats(network_dir, binary, thresh)
    l_names, l_graphs = _parse_edgelists(edge_lists)
    return nd.MultilayerNetwork.from_nx_layers(network_list=l_graphs, layer_names=l_names)
