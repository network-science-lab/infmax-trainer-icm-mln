import logging
import os
from typing import Any
from unittest.mock import MagicMock

import neptune
from lightning.pytorch import loggers
from lightning.pytorch.loggers.logger import Logger


class DummyLogger:
    def __init__(*args, **kwargs):
        pass

    def __getattr__(self, name):
        return type(self)


def get_lightning_neptune(config: dict[str, Any]) -> loggers.NeptuneLogger | MagicMock:
    """Initialise neptune logger integrated with Lightning or return MagicMock."""
    logging.getLogger("neptune").setLevel(logging.CRITICAL)
    try:
        logger = loggers.NeptuneLogger(
            api_key=os.getenv(
                key="NEPTUNE_API_KEY",
                default=neptune.ANONYMOUS_API_TOKEN,
            ),
            project="infmax/infmax-gnn",
            tags=[
                tag_config["name"]
                for tag_config in config["training"]["logger"]["tags"]
            ],
            description=config["model"]["name"],
            name=config["model"]["name"],
        )
    except Exception as e:
        logging.exception(e)
        logging.warning("Neptune not initialised - using mocked logger!")
        logger = DummyLogger()

    return logger


def get_loggers(config: dict[str, Any]) -> Logger:
    match config["training"]["logger"]["name"]:
        case "tensor_board":
            return loggers.TensorBoardLogger(
                save_dir=config["hydra"]["run"]["dir"],
                name="tensorboard",
            )
        case "neptune":
            return get_lightning_neptune(config)
        case _:
            logging.warning(f"{config['training']['loggers']['name']} is not supported")
            logging.info("Using mocked logger")
            return DummyLogger()
