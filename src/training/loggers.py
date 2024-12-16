import logging
import os
from typing import Any
from unittest.mock import MagicMock

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
            logging.getLogger("neptune").setLevel(logging.CRITICAL)
            try:
                raise Exception("aaa")
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
            except Exception as e:
                logging.info(e)
                logging.info("Neptune not initialized - using mocked logger!")
                logger = MagicMock()
            return logger
        case _:
            logging.warning(f"{config['training']['loggers']['name']} is not supported")
