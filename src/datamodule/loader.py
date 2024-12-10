import logging

from typing import Any, Literal

from _data_set.nsl_data_utils.loaders.net_loader import load_network
from _data_set.nsl_data_utils.loaders.sp_loader import load_sp
from sklearn.model_selection import train_test_split
from src import MODULE_PATH
from src.dataset.base_hetero_dataset import BaseHeteroDataset
from src.dataset.data_frame_hetero_dataset import DataFrameHeteroDataset
from src.data_models.mln_info import MLNInfo
from src.utils.worker import get_num_workers
from torch_geometric.data.lightning import LightningDataset
from torch_geometric.typing import EdgeType, NodeType


def _load_mln_info_chunk(
    network_type: str,
    labels_type: str,
    features_type: str,
    protocol: str,
    p_value: float,
    random_seed: int,
    dataset_type: Literal["train", "val", "test"],
    validation_split: bool,
) -> list[MLNInfo]:
    """Load raw networks and target labels for given network type and spreading params."""

    nets_dict = load_network(net_name=network_type, as_tensor=False)
    sps_dict = load_sp(net_name=network_type)
    nets_to_use = list(nets_dict.keys())
    assert len(nets_dict) == len(sps_dict)

    if validation_split: # if we make a split here, loading is speeded up
        train_nets, val_nets = train_test_split(
            nets_to_use, test_size=0.2, random_state=random_seed
        )
        if dataset_type == "train":
            nets_to_use = train_nets
        elif dataset_type == "val":
            nets_to_use = val_nets
        else:
            raise ValueError(f"Invalid option: val_split for the test network!")

    mln_info = [  # for the reduced size of data, we create shorter list of MNI objects
        MLNInfo(
            mln_type=network_type,
            mln_name=net_name,
            mln=nets_dict[net_name],
            icm_protocol=protocol,
            icm_p=p_value,
            x_type=features_type,
            y_type=labels_type,
            sp_raw=sps_dict[net_name],
        )
        for net_name in nets_to_use
    ]

    return mln_info


def _get_dataset(
    data_name: str,
    networks_config: list[dict[str, Any]],
    labels: list[str],
    protocol: str,
    p_value: float,
    random_seed: int,
    input_dim: int,
    output_dim: int,
    dataset_type: Literal["train", "val", "test"],
) -> BaseHeteroDataset:
    logging.info(f"Loading {dataset_type} dataset.")
    match data_name:
        case DataFrameHeteroDataset.__name__:
            mlni_nets = []
            for network_config in networks_config:
                mlni_chunk = _load_mln_info_chunk(
                    network_type=network_config["name"],
                    labels_type=labels,
                    features_type=network_config["features_type"],
                    protocol=protocol,
                    p_value=p_value,
                    random_seed=random_seed,
                    dataset_type=dataset_type,
                    validation_split=bool(network_config.get("val_split")),
                )
                mlni_nets.extend(mlni_chunk)
            return DataFrameHeteroDataset(
                root=str(MODULE_PATH.parent / "data"),  # TODO: this path doesn't exist
                networks=mlni_nets,
                input_dim=input_dim,
                output_dim=output_dim,
            )
        case _:
            raise AttributeError(f"Unknown dataset: {data_name}")


def get_datasets(config: dict[str, Any]) -> dict[str, BaseHeteroDataset]:
    train_dataset = _get_dataset(
        data_name=config["data"]["name"],
        networks_config=config["data"]["train_networks"],
        labels=config["data"]["output_label_name"],
        input_dim=config["model"]["parameters"]["input_dim"],
        output_dim=config["model"]["parameters"]["output_dim"],
        protocol=config["data"]["protocol"],
        p_value=config["data"]["p_value"],
        random_seed=config["base"]["random_seed"],
        dataset_type="train",
    )
    val_dataset = _get_dataset(
        data_name=config["data"]["name"],
        networks_config=config["data"]["val_dataset"],
        labels=config["data"]["output_label_name"],
        input_dim=config["model"]["parameters"]["input_dim"],
        output_dim=config["model"]["parameters"]["output_dim"],
        protocol=config["data"]["protocol"],
        p_value=config["data"]["p_value"],
        random_seed=config["base"]["random_seed"],
        dataset_type="val",
    )
    test_dataset = _get_dataset(
        data_name=config["data"]["name"],
        networks_config=config["data"]["test_dataset"],
        labels=config["data"]["output_label_name"],
        input_dim=config["model"]["parameters"]["input_dim"],
        output_dim=config["model"]["parameters"]["output_dim"],
        protocol=config["data"]["protocol"],
        p_value=config["data"]["p_value"],
        random_seed=config["base"]["random_seed"],
        dataset_type="test",
    )
    return {
        "train": train_dataset,
        "val": val_dataset,
        "test": test_dataset
    }


def get_datamodule(
    datasets: dict[str, BaseHeteroDataset], config: dict[str, Any]
) -> LightningDataset:
    return LightningDataset(
        train_dataset=datasets["train"].data_list,
        val_dataset=datasets["val"].data_list,
        test_dataset=datasets["test"].data_list,
        batch_size=config["data"]["batch"]["gradient_accumulation_step"],
        num_workers=get_num_workers(config),
    )


def get_metadata(datasets: list[BaseHeteroDataset]) -> tuple[list[NodeType], list[EdgeType]]:
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
