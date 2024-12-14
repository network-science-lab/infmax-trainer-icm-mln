from pathlib import Path
from typing import Any

import yaml

from hydra.core.hydra_config import HydraConfig
from omegaconf import DictConfig
from _data_set.nsl_data_utils.loaders.constants import CENTRALITY_FUNCTIONS
from src import CONFIGS_PATH


def load_config(
    cofig_path: str | Path,
    cfg: DictConfig | None = None,
) -> dict[str, Any]:
    with open(cofig_path, "r") as file:
        config = yaml.safe_load(file)

    if cfg:
        config["default"] = cfg
    config["hydra"] = HydraConfig.get()

    return config


def get_available_configs() -> list[str]:
    return [
        config_path.stem
        for config_path in CONFIGS_PATH.iterdir()
        if config_path.stem != "hydra"
    ]


def validate_config(args: dict[str, Any]) -> None:
    """Check whether input and output dimensions for model match the dataset."""
    model_params = args["model"]["parameters"]
    data_params = args["data"]
    for train_data in data_params["train_data"]:
        if train_data["features_type"] == "centralities":
            assert model_params["input_dim"] <= len(CENTRALITY_FUNCTIONS)
            continue
    for train_data in data_params["test_data"]:
        if train_data["features_type"] == "centralities":
            assert model_params["input_dim"] <= len(CENTRALITY_FUNCTIONS)
            continue
    assert model_params["output_dim"] == len(data_params["output_label_name"])
