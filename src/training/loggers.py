import logging
import os
from typing import Any

import neptune
from lightning.pytorch import loggers
from lightning.pytorch.loggers.logger import Logger


def get_loggers(config: dict[str, Any]) -> Logger:
    match config["training"]["logger"]["name"]:
        case "tensor_board":
            return loggers.TensorBoardLogger(
                save_dir=config["hydra"]["run"]["dir"],
                name="tensorboard",
            )
        case "neptune":
            logger = loggers.NeptuneLogger(
                api_key=os.getenv(
                    key="NEPTUNE_API_KEY",
                    default=neptune.ANONYMOUS_API_TOKEN,
                ),
                project="infmax/infmax-gnn",
                tags=[tag_config["name"] for tag_config in config["training"]["logger"]["tags"]],
                description=config["model"]["name"],
                name=config["model"]["name"],
            )
            logging.getLogger("neptune").setLevel(logging.CRITICAL)
            return logger
        case _:
            logging.warning(f"{config['training']['loggers']['name']} is not supported")




# for torch trainer

from typing import Dict, Union
import neptune.new
import neptune
from unittest.mock import MagicMock


def init_neptune(config: Dict) -> Union[MagicMock, neptune.Run]:
    try:
        logging.getLogger("neptune").setLevel(logging.CRITICAL)
        neptune_instance = create_neptune_instance(config)
    except Exception as e:
        logging.info(e)
        logging.info("Neptune not initialized - using mocked logger!")
        neptune_instance = MagicMock()
    return neptune_instance


def create_neptune_instance(config: dict[str, Any]):
    neptune_instance = neptune.init_run(
        project="infmax/infmax-gnn",
        api_token=os.getenv(
            key="NEPTUNE_API_KEY",
            default=neptune.ANONYMOUS_API_TOKEN,
        ),
        name=config["model"]["name"],
        tags=[tag_config["name"] for tag_config in config["training"]["logger"]["tags"]],
    )
    return neptune_instance
