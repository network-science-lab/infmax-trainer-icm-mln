import logging

from typing import Any, Literal

import torch
import torch.utils
import torch.utils.data

from torch_geometric.data.lightning import LightningDataset
from torch_geometric.typing import EdgeType, NodeType

from _data_set.nsl_data_utils.loaders.net_loader import load_net_names
from _data_set.nsl_data_utils.loaders.sp_loader import load_sp_paths
from src import MODULE_PATH
from src.data_models.mln_info import MLNInfo
from src.data_sets.base_dataset import BaseDataSet
from src.data_sets.super_spreaders_dataset import SuperSpreadersDataSet
from src.utils.worker import get_num_workers


def _load_mln_info_chunk(
    network_type: str,
    labels_type: str,
    features_type: str,
    protocol: str,
    p_value: float,
) -> list[MLNInfo]:
    """Load raw networks and target labels for given network type and spreading params."""
    mlni_chunk = []
    for net_name in load_net_names(net_type=network_type):
        mlni_chunk.append(
            MLNInfo.from_config(
                mln_type=network_type,
                mln_name=net_name,
                icm_protocol=protocol,
                icm_p=p_value,
                x_type=features_type,
                y_type=labels_type,
                sp_paths=load_sp_paths(net_type=network_type, net_name=net_name),
            )
        )
    return mlni_chunk


# TODO: remove input_dim, output_dim from config
def _get_dataset(
    data_name: str,
    networks_config: list[dict[str, Any]],
    labels: list[str],
    protocol: str,
    p_value: float,
    # random_seed: int,
    # input_dim: int,
    # output_dim: int,
    # dataset_type: Literal["train", "val", "test"],
) -> BaseDataSet:
    match data_name:
        case SuperSpreadersDataSet.__name__:
            mlni_nets = []
            for network_config in networks_config:
                mlni_chunk = _load_mln_info_chunk(
                    network_type=network_config["name"],
                    labels_type=labels,
                    features_type=network_config["features_type"],
                    protocol=protocol,
                    p_value=p_value,
                )
                mlni_nets.extend(mlni_chunk)
            return SuperSpreadersDataSet(
                root=str(MODULE_PATH.parent / "_data_set/nsl_data_sources"),
                networks=mlni_nets,
                # input_dim=input_dim,
                # output_dim=output_dim,
            )
        case _:
            raise AttributeError(f"Unknown dataset: {data_name}")


def get_datasets(config: dict[str, Any]) -> dict[str, BaseDataSet]:
    logging.info(f"Loading train dataset.")
    dataset = _get_dataset(
        data_name=config["data"]["name"],
        networks_config=config["data"]["train_data"],
        labels=config["data"]["output_label_name"],
        # input_dim=config["model"]["parameters"]["input_dim"],
        # output_dim=config["model"]["parameters"]["output_dim"],
        protocol=config["data"]["protocol"],
        p_value=config["data"]["p_value"],
        # random_seed=config["base"]["random_seed"],
    )
    logging.info(f"Splitting to train/eval dataset.")
    val_len = int(len(dataset) * config["data"]["train_data"]["val_ratio"])
    train_len = len(dataset) - val_len
    train_dataset, val_dataset = torch.utils.data.random_split(train_dataset, [train_len, val_len])  # TODO: check repeitiveness
    logging.info(f"Loading test dataset.")
    test_dataset = _get_dataset(
        data_name=config["data"]["name"],
        networks_config=config["data"]["test_dataset"],
        labels=config["data"]["output_label_name"],
        # input_dim=config["model"]["parameters"]["input_dim"],
        # output_dim=config["model"]["parameters"]["output_dim"],
        protocol=config["data"]["protocol"],
        p_value=config["data"]["p_value"],
        # random_seed=config["base"]["random_seed"],
    )
    return {
        "train": train_dataset,
        "val": val_dataset,
        "test": test_dataset
    }


def get_datamodule(
    datasets: dict[str, BaseDataSet], config: dict[str, Any]
) -> LightningDataset:
    return LightningDataset(
        train_dataset=datasets["train"].data_list,
        val_dataset=datasets["val"].data_list,
        test_dataset=datasets["test"].data_list,
        batch_size=config["data"]["batch"]["gradient_accumulation_step"],
        num_workers=get_num_workers(config),
        # pin_memory=False,
    )


def get_metadata(datasets: list[BaseDataSet]) -> tuple[list[NodeType], list[EdgeType]]:
    """
    Here, we treat as metadata types of relations and types of agents.

    In current approach it's just "actors" for agents and a union of layer names available in the 
    dataset. This function is used only in autoconverters from base torch_geometric models
    to heterogeneous ones.
    """
    nodes_data = set()
    edges_data = set()

    for dataset in datasets:
        node_data, edge_data = dataset.get_metadata()

        nodes_data = nodes_data.union(node_data)
        edges_data = edges_data.union(edge_data)

    return list(nodes_data), list(edges_data)
