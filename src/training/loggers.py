import logging
from typing import Any

from lightning.pytorch import loggers
from lightning.pytorch.loggers.logger import Logger


def get_loggers(config: dict[str, Any]) -> list[Logger]:
    result = []

    for logger_config in config["training"]["loggers"]:
        match logger_config["name"]:
            case "tensor_board":
                result.append(
                    loggers.TensorBoardLogger(
                        save_dir=config["hydra"]["run"]["dir"],
                        name="tensorboard",
                    )
                )
            case _:
                logging.warning(f"{logger_config['name']} is not supported")

    return result
