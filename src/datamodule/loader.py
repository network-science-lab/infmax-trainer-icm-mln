from typing import Any, Literal

from _data_set.nsl_data_utils.loaders.net_loader import load_network
from _data_set.nsl_data_utils.loaders.sp_loader import load_sp
from sklearn.model_selection import train_test_split
from src import MODULE_PATH
from src.dataset.base_hetero_dataset import BaseHeteroDataset
from src.dataset.data_frame_hetero_dataset import DataFrameHeteroDataset
from src.utils.multilayer_network import MultilayerNetworkInfo
from src.utils.worker import get_num_workers
from torch_geometric.data.lightning import LightningDataset
from torch_geometric.typing import EdgeType, NodeType


def _load_networks_info(
    networks_config: list[dict[str, Any]],
    label: str,
    protocol: str,
    random_seed: int,
    dataset_type: Literal["train", "val", "test"],
) -> list[MultilayerNetworkInfo]:
    networks_info = []
    for network_config in networks_config:
        network_name = network_config["name"]
        network = load_network(
            net_name=network_name,
            as_tensor=False,
        )
        spreading_potential = load_sp(network_name)

        network_info = [
            MultilayerNetworkInfo(
                network=mln,
                output_label_name=label,
                spreading_potential=spreading_potential[name],
                network_name=f"{network_name}_{name}",
                protocol=protocol,
                features_type=network_config["features_type"],
            )
            for name, mln in network.items()
        ]

        if "val_split" in network_config and network_config["val_split"]:
            train_networks, val_networks = train_test_split(
                network_info,
                test_size=0.2,
                random_state=random_seed,
            )
            match dataset_type:
                case "train":
                    networks_info.extend(train_networks)
                case "val":
                    networks_info.extend(val_networks)
                case "test":
                    raise ValueError(f"Invalid option: val_split for test network")
        else:
            networks_info.extend(network_info)

    return networks_info


def _get_dataset(
    data_name: str,
    networks_config: list[dict[str, Any]],
    label: str,
    input_dim: int,
    output_dim: int,
    protocol: str,
    random_seed: int,
    dataset_type: Literal["train", "val", "test"],
) -> BaseHeteroDataset:
    match data_name:
        case DataFrameHeteroDataset.__name__:
            return DataFrameHeteroDataset(
                root=str(MODULE_PATH.parent / "data"),
                networks=_load_networks_info(
                    networks_config=networks_config,
                    label=label,
                    protocol=protocol,
                    random_seed=random_seed,
                    dataset_type=dataset_type,
                ),
                input_dim=input_dim,
                output_dim=output_dim,
            )

        case _:
            raise AttributeError(f"Unknown dataset: {data_name}")


def get_datasets(config: dict[str, Any]) -> dict[str, BaseHeteroDataset]:
    train_dataset = _get_dataset(
        data_name=config["data"]["name"],
        networks_config=config["data"]["train_networks"],
        label=config["data"]["output_label_name"],
        input_dim=config["model"]["parameters"]["input_dim"],
        output_dim=config["model"]["parameters"]["output_dim"],
        protocol=config["data"]["protocol"],
        random_seed=config["base"]["random_seed"],
        dataset_type="train",
    )
    val_dataset = _get_dataset(
        data_name=config["data"]["name"],
        networks_config=config["data"]["val_dataset"],
        label=config["data"]["output_label_name"],
        input_dim=config["model"]["parameters"]["input_dim"],
        output_dim=config["model"]["parameters"]["output_dim"],
        protocol=config["data"]["protocol"],
        random_seed=config["base"]["random_seed"],
        dataset_type="val",
    )
    test_dataset = _get_dataset(
        data_name=config["data"]["name"],
        networks_config=config["data"]["test_dataset"],
        label=config["data"]["output_label_name"],
        input_dim=config["model"]["parameters"]["input_dim"],
        output_dim=config["model"]["parameters"]["output_dim"],
        protocol=config["data"]["protocol"],
        random_seed=config["base"]["random_seed"],
        dataset_type="test",
    )

    return {
        "train": train_dataset,
        "val": val_dataset,
        "test": test_dataset,
    }


def get_datamodule(
    datasets: dict[str, BaseHeteroDataset],
    config: dict[str, Any],
) -> LightningDataset:
    return LightningDataset(
        train_dataset=datasets["train"].data_list,
        val_dataset=datasets["val"].data_list,
        test_dataset=datasets["test"].data_list,
        batch_size=config["data"]["real_batch_size"],
        num_workers=get_num_workers(config),
    )


def get_metadata(
    datasets: list[BaseHeteroDataset],
) -> tuple[list[NodeType], list[EdgeType]]:
    nodes_data = set()
    edges_data = set()

    for dataset in datasets:
        node_data, edge_data = dataset.get_metadata()

        nodes_data = nodes_data.union(node_data)
        edges_data = edges_data.union(edge_data)

    return list(nodes_data), list(edges_data)
