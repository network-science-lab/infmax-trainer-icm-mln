import logging
from typing import Any

import torch
import torch.utils
import torch.utils.data
from torch.utils.data.dataset import Subset
from torch_geometric.data.lightning import LightningDataset

from data.tsds_utils.loaders.net_loader import load_net_names
from src.data_models.mln_info import MLNInfo
from src.dataset import transforms
from src.dataset.super_spreaders_dataset import SuperSpreadersDataset

TEST_ARTIFICIAL_NETWORKS = {
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
    tr_params = (
        transform["parameters"] if isinstance(transform["parameters"], dict) else {}
    )
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


def _train_val_test_split(
    config: dict[str, Any],
    dataset: SuperSpreadersDataset,
) -> tuple[Subset, Subset, SuperSpreadersDataset]:
    train_val_mlninfos, test_mlninfos = [], []
    for mlninfo in dataset.data_list:
        if (mlninfo.mln_type, mlninfo.mln_name) in TEST_ARTIFICIAL_NETWORKS:
            test_mlninfos.append(mlninfo)
        else:
            train_val_mlninfos.append(mlninfo)

    train_val_dataset = SuperSpreadersDataset(
        networks=train_val_mlninfos,
        input_dim=dataset._input_dim,
        output_dim=dataset._output_dim,
        transform=dataset.transform,
    )

    val_len = int(len(dataset) * config["data"]["val_data_ratio"])
    train_len = len(dataset) - val_len - len(test_mlninfos)
    train_dataset, val_dataset = torch.utils.data.random_split(
        dataset=train_val_dataset,
        lengths=[train_len, val_len],
        generator=torch.Generator().manual_seed(config["base"]["random_seed"]),
    )

    return (
        train_dataset,
        val_dataset,
        SuperSpreadersDataset(
            networks=test_mlninfos,
            input_dim=dataset._input_dim,
            output_dim=dataset._output_dim,
            transform=dataset.transform,
        ),
    )


def get_datasets(config: dict[str, Any]) -> dict[str, SuperSpreadersDataset]:
    logging.info(f"Loading dataset (paths).")
    dataset = get_dataset(
        data_name=config["data"]["name"],
        networks_config=config["data"]["train_data"],
        labels=config["data"]["output_label_name"],
        protocol=config["data"]["icm"]["protocol"],
        p=config["data"]["icm"]["p"],
        input_dim=config["model"]["parameters"]["input_dim"],
        transform=config["data"]["transform"],
    )
    logging.info(f"Splitting to train/eval/test dataset (paths).")

    train_dataset, val_dataset, test_dataset = _train_val_test_split(
        config=config,
        dataset=dataset,
    )

    logging.info(f"Loading test dataset (paths).")
    if config["data"].get("test_data"):
        _test_dataset = get_dataset(
            data_name=config["data"]["name"],
            networks_config=config["data"]["test_data"],
            labels=config["data"]["output_label_name"],
            protocol=config["data"]["icm"]["protocol"],
            p=config["data"]["icm"]["p"],
            input_dim=config["model"]["parameters"]["input_dim"],
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
