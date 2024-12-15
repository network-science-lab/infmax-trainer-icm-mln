import gc
import logging
from typing import Any

import torch
from pytorch_lightning import callbacks, Callback, LightningModule

# I've tried this, but it doesn't work
# class MemoryReleaserCallback(Callback):

#     def _clear_memory(self) -> None:
#         torch.cuda.empty_cache()
#         gc.collect()

#     def on_train_epoch_end(self, trainer, pl_module: LightningModule):
#         pl_module.log(f"{self.__class__.__name__}: releasing memory [train]!", 0)
#         logging.critical(f"{self.__class__.__name__}: releasing memory [train]!")
#         self._clear_memory()

#     def on_validation_epoch_end(self, trainer, pl_module):
#         logging.critical(f"{self.__class__.__name__}: releasing memory [val]!")
#         self._clear_memory()

#     def on_test_epoch_end(self, trainer, pl_module):
#         logging.critical(f"{self.__class__.__name__}: releasing memory [test]!")
#         self._clear_memory()


def get_callbacks(config: dict[str, Any]) -> list[callbacks.Callback]:
    result = []

    for callback_config in config["training"]["callbacks"]:
        match callback_config["name"]:
            case "model_summary":
                result.append(
                    callbacks.ModelSummary(max_depth=callback_config["max_depth"])
                )
            case "model_checkpoint":
                result.append(
                    callbacks.ModelCheckpoint(
                        dirpath=config["hydra"]["run"]["dir"],
                        mode=callback_config["mode"],
                        monitor=callback_config["monitor"],
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
        # result.append(MemoryReleaserCallback())

    return result
