from pathlib import Path
from typing import Any

import yaml
from hydra.core.hydra_config import HydraConfig
from omegaconf import DictConfig


def load_config(
    cfg: DictConfig,
    cofig_path: str | Path,
) -> dict[str, Any]:
    with open(cofig_path, "r") as file:
        config = yaml.safe_load(file)

    config["default"] = cfg
    config["hydra"] = HydraConfig.get()

    return config

from src import CONFIGS_PATH

def get_available_configs() -> list[str]:
    return [config_path.stem for config_path in CONFIGS_PATH.iterdir() if config_path.stem != "hydra"]
