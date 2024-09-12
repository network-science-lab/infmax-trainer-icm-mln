import logging
from typing import Any

from pytorch_lightning import callbacks


def get_callbacks(config: dict[str, Any]) -> list[callbacks.Callback]:
    result = []

    for callback_config in config["training"]["callbacks"]:
        match callback_config["name"]:
            case "model_checkpoint":
                result.append(
                    callbacks.ModelCheckpoint(
                        dirpath=config["hydra"]["run"]["dir"],
                        mode=callback_config["mode"],
                        save_last=callback_config["save_last"],
                        save_top_k=callback_config["save_top_k"],
                        verbose=callback_config["verbose"],
                    )
                )
            case "early_stopping":
                result.append(
                    callbacks.EarlyStopping(
                        monitor=callback_config["monitor"],
                        mode=callback_config["mode"],
                        patience=callback_config["patience"],
                    )
                )
            case "gradient_accumulation_scheduler":
                gradient_scheduling = callback_config["gradient_scheduling"]
                scheduling = (
                    gradient_scheduling
                    if gradient_scheduling is dict
                    else {0: gradient_scheduling}
                )
                result.append(
                    callbacks.GradientAccumulationScheduler(scheduling=scheduling)
                )
            case _:
                logging.warning(f"{callback_config['name']} is not supported")

    return result
