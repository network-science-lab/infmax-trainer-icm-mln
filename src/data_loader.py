import logging

from typing import Any

import torch
import torch.utils
import torch.utils.data

from torch_geometric.data.lightning import LightningDataset
from torch_geometric.typing import EdgeType, NodeType

from _data_set.nsl_data_utils.loaders.net_loader import load_net_names
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
            )
        )
    return mlni_chunk


def _get_dataset(
    data_name: str,
    networks_config: list[dict[str, Any]],
    labels: list[str],
    protocol: str,
    p_value: float,
    input_dim: int,
    output_dim: int,
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
            return SuperSpreadersDataSet(mlni_nets, input_dim, output_dim)
        case _:
            raise AttributeError(f"Unknown dataset: {data_name}")


def get_datasets(config: dict[str, Any]) -> dict[str, BaseDataSet]:
    logging.info(f"Loading train dataset.")
    dataset = _get_dataset(
        data_name=config["data"]["name"],
        networks_config=config["data"]["train_data"],
        labels=config["data"]["output_label_name"],
        input_dim=config["model"]["parameters"]["input_dim"],
        output_dim=config["model"]["parameters"]["output_dim"],
        protocol=config["data"]["icm"]["protocol"],
        p_value=config["data"]["icm"]["p"],
    )
    logging.info(f"Splitting to train/eval dataset.")
    val_len = int(len(dataset) * config["data"]["val_data"]["ratio"])
    train_len = len(dataset) - val_len
    train_dataset, val_dataset = torch.utils.data.random_split(
        dataset=dataset,
        lengths=[train_len, val_len],
        generator=torch.Generator().manual_seed(config["base"]["random_seed"])
    )
    logging.info(f"Loading test dataset.")
    test_dataset = _get_dataset(
        data_name=config["data"]["name"],
        networks_config=config["data"]["test_data"],
        labels=config["data"]["output_label_name"],
        input_dim=config["model"]["parameters"]["input_dim"],
        output_dim=config["model"]["parameters"]["output_dim"],
        protocol=config["data"]["icm"]["protocol"],
        p_value=config["data"]["icm"]["p"],
    )
    logging.info(f"Graphs: test-{len(train_dataset)}, eval-{len(val_dataset)}, test-{len(test_dataset)}")
    return {"train": train_dataset, "val": val_dataset, "test": test_dataset}


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
