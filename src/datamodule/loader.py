import logging
from typing import Any

import torch
import torch.utils
import torch.utils.data
from torch_geometric.data.lightning import LightningDataset
from torch_geometric.typing import EdgeType, NodeType

from _data_set.nsl_data_utils.loaders.net_loader import load_net_names
from src.data_models.mln_info import MLNInfo
from src.dataset import transforms
from src.dataset.super_spreaders_dataset import SuperSpreadersDataset


CLEAN_AFTER_ME_PLEASE = {
    ("artificial_er", "network_15"),
    ("artificial_er", "network_20"),
    ("artificial_er", "network_40"),
    ("artificial_er", "network_45"),
    ("artificial_er", "network_71"),
    ("artificial_er", "network_78"),
    ("artificial_er", "network_79"),
    ("artificial_er", "network_80"),
    ("artificial_pa", "network_7"),
    ("artificial_pa", "network_23"),
    ("artificial_pa", "network_39"),
    ("artificial_pa", "network_57"),
    ("artificial_pa", "network_58"),
    ("artificial_pa", "network_75"),
    ("artificial_pa", "network_85"),
    ("artificial_pa", "network_95"),
    ("artificial_pa", "network_22"),
    ("artificial_pa", "network_68"),
    ("artificial_pa", "network_83"),
    ("artificial_pa", "network_93"),
}


def _load_mln_info_chunk(
    network_type: str,
    labels_type: str,
    features_type: str,
    protocol: str,
    p: float,
) -> list[MLNInfo]:
    """Load raw networks and target labels for given network type and spreading params."""
    mlni_chunk = [
        MLNInfo.from_config(
            mln_type=network_type,
            mln_name=net_name,
            icm_protocol=protocol,
            icm_p=p,
            x_type=features_type,
            y_type=labels_type,
        )
        for net_name in load_net_names(net_type=network_type)
    ]

    return mlni_chunk


def get_transform(transform: str):
    """Get data transformation according to provided configuration."""
    tr_class = getattr(transforms, transform["name"], None)
    tr_params = transform["parameters"] if isinstance(transform["parameters"], dict) else {}
    if tr_class:
        return tr_class(**tr_params)
    else:
        return None


def get_dataset(
    data_name: str,
    networks_config: list[dict[str, Any]],
    labels: list[str],
    protocol: str,
    p: float,
    input_dim: int,
    transform: str,
) -> SuperSpreadersDataset:
    if data_name == SuperSpreadersDataset.__name__:
        mlni_nets = []
        for network_config in networks_config:
            mlni_chunk = _load_mln_info_chunk(
                network_type=network_config["name"],
                labels_type=labels,
                features_type=network_config["features_type"],
                protocol=protocol,
                p=p,
            )
            mlni_nets.extend(mlni_chunk)
        return SuperSpreadersDataset(
            networks=mlni_nets,
            input_dim=input_dim,
            output_dim=len(labels),
            transform=get_transform(transform),
        )
    raise AttributeError(f"Unknown dataset: {data_name}")


def get_datasets(config: dict[str, Any]) -> dict[str, SuperSpreadersDataset]:
    logging.info(f"Loading dataset (paths).")
    dataset = get_dataset(
        data_name=config["data"]["name"],
        networks_config=config["data"]["train_data"],
        labels=config["data"]["output_label_name"],
        protocol=config["data"]["icm"]["protocol"],
        p=config["data"]["icm"]["p"],
        input_dim=config["model"]["parameters"]["input_dim"],  # TODO: a convolved parameter!
        transform=config["data"]["transform"],
    )
    logging.info(f"Splitting to train/eval/test dataset (paths).")
    val_len = int(len(dataset) * config["data"]["val_data_ratio"])
    ##### ##### ##### dirty deletion - start
    # test_len = int(len(dataset) * config["data"]["test_data_ratio"])
    # train_len = len(dataset) - val_len - test_len
    # train_dataset, val_dataset, test_dataset = torch.utils.data.random_split( # TODO: handle the test data provided explicitly
    #     dataset=dataset,
    #     lengths=[train_len, val_len, test_len],
    #     generator=torch.Generator().manual_seed(config["base"]["random_seed"]),
    # )
    ##### ##### ##### dirty deletion - stop
    ##### ##### ##### dirty addition - start
    train_val_mlninfos, test_mlninfos = [], []
    for mlninfo in dataset.data_list:
        mlninfo_id = (mlninfo.mln_type, mlninfo.mln_name)
        if mlninfo_id in CLEAN_AFTER_ME_PLEASE:
            test_mlninfos.append(mlninfo)
        else:
            train_val_mlninfos.append(mlninfo)
    test_dataset = SuperSpreadersDataset(
        networks=test_mlninfos,
        input_dim=dataset._input_dim,
        output_dim=dataset._output_dim,
        transform=dataset.transform,
    )
    train_val_dataset = SuperSpreadersDataset(
        networks=train_val_mlninfos,
        input_dim=dataset._input_dim,
        output_dim=dataset._output_dim,
        transform=dataset.transform,
    )
    train_len = len(dataset) - val_len - len(test_dataset)
    train_dataset, val_dataset = torch.utils.data.random_split(
        dataset=train_val_dataset,
        lengths=[train_len, val_len],
        generator=torch.Generator().manual_seed(config["base"]["random_seed"]),
    )
    ##### ##### ##### dirty addition - stop
    logging.info(f"Loading test dataset (paths).")
    if config['data'].get("test_data"):
        _test_dataset = get_dataset(
            data_name=config["data"]["name"],
            networks_config=config["data"]["test_data"],
            labels=config["data"]["output_label_name"],
            protocol=config["data"]["icm"]["protocol"],
            p=config["data"]["icm"]["p"],
            input_dim=config["model"]["parameters"]["input_dim"],  # TODO: a convolved parameter!
            transform=config["data"]["transform"],
        )
        test_dataset = torch.utils.data.ConcatDataset([test_dataset, _test_dataset])

    logging.info(
        f"Graphs: test-{len(train_dataset)}, eval-{len(val_dataset)}, test-{len(test_dataset)}"
    )
    return {
        "train": train_dataset,
        "val": val_dataset,
        "test": test_dataset,
    }


def get_datamodule(
    datasets: dict[str, SuperSpreadersDataset], config: dict[str, Any]
) -> LightningDataset:
    return LightningDataset(
        train_dataset=datasets["train"],
        val_dataset=datasets["val"],
        test_dataset=datasets["test"],
        batch_size=1,
        num_workers=0,
        pin_memory=True,
    )


def get_metadata(
    datasets: list[SuperSpreadersDataset],
) -> tuple[list[NodeType], list[EdgeType]]:
    """
    Here, we treat as metadata types of relations and types of agents.

    In current approach it's just "actors" for agents and a union of layer names available in the
    dataset. This function is used only in autoconverters from base torch_geometric models
    to heterogeneous ones.
    """
    nodes_data = set()
    edges_data = set()

    # for dataset in datasets: # TODO:
    #     node_data, edge_data = dataset.get_metadata()

    #     nodes_data = nodes_data.union(node_data)
    #     edges_data = edges_data.union(edge_data)

    return list(nodes_data), list(edges_data)
