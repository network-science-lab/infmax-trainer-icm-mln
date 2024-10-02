import logging
import os
from typing import Any

import neptune
from lightning.pytorch import loggers
from lightning.pytorch.loggers.logger import Logger
from pytorch_lightning import LightningModule


def get_loggers(
    config: dict[str, Any],
    model: LightningModule,
) -> list[Logger]:
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
            case "neptune":
                logger = loggers.NeptuneLogger(
                    api_key=os.getenv(
                        key="NEPTUNE_API_KEY",
                        default=neptune.ANONYMOUS_API_TOKEN,
                    ),
                    project="infmax/infmax-gnn",
                    tags=[tag_config["name"] for tag_config in logger_config["tags"]],
                    description=config["model"]["name"],
                    name=config["model"]["name"],
                )

                logger.log_model_summary(
                    model=model,
                    max_depth=logger_config["model_summary_max_depth"],
                )
                logger.log_hyperparams(
                    {key: value for key, value in config.items() if key != "hydra"}
                )

                result.append(logger)
                logging.getLogger("neptune").setLevel(logging.CRITICAL)
            case _:
                logging.warning(f"{logger_config['name']} is not supported")

    return result
