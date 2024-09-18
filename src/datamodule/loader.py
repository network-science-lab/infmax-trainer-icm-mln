from typing import Any

from torch_geometric.data.lightning import LightningDataset
from torch_geometric.typing import EdgeType, NodeType

from _data_set.nsl_data_utils.loaders.net_loader import load_network
from _data_set.nsl_data_utils.loaders.sp_loader import load_sp
from src import MODULE_PATH
from src.dataset.data_frame_hetero_dataset import DataFrameHeteroDataset
from src.utils.multilayer_network import MultilayerNetworkInfo
from src.utils.worker import get_num_workers


def _get_dataset(
    networks_config: list[dict[str, Any]],
    label: str,
    input_dim: int,
    output_dim: int,
) -> DataFrameHeteroDataset:
    return DataFrameHeteroDataset(
        root=str(MODULE_PATH.parent / "data"),
        networks=[
            MultilayerNetworkInfo(
                network=load_network(
                    net_name=network_config["name"],
                    as_tensor=False,
                ),
                output_label_name=label,
                spreading_potential=load_sp(network_config["name"]),
                network_name=network_config["name"],
            )
            for network_config in networks_config
        ],
        input_dim=input_dim,
        output_dim=output_dim,
    )


def get_datasets(config: dict[str, Any]) -> dict[str, DataFrameHeteroDataset]:
    train_dataset = _get_dataset(
        networks_config=config["data"]["train_networks"],
        label=config["data"]["label_name"],
        input_dim=config["model"]["parameters"]["input_dim"],
        output_dim=config["model"]["parameters"]["output_dim"],
    )
    val_dataset = _get_dataset(
        networks_config=config["data"]["val_dataset"],
        label=config["data"]["label_name"],
        input_dim=config["model"]["parameters"]["input_dim"],
        output_dim=config["model"]["parameters"]["output_dim"],
    )
    test_dataset = _get_dataset(
        networks_config=config["data"]["test_dataset"],
        label=config["data"]["label_name"],
        input_dim=config["model"]["parameters"]["input_dim"],
        output_dim=config["model"]["parameters"]["output_dim"],
    )

    return {
        "train": train_dataset,
        "val": val_dataset,
        "test": test_dataset,
    }


def get_datamodule(
    datasets: dict[str, DataFrameHeteroDataset],
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
    datasets: list[DataFrameHeteroDataset],
) -> tuple[list[NodeType], list[EdgeType]]:
    nodes_data = set()
    edges_data = set()

    for dataset in datasets:
        node_data, edge_data = dataset.get_metadata()

        nodes_data = nodes_data.union(node_data)
        edges_data = edges_data.union(edge_data)

    return list(nodes_data), list(edges_data)
