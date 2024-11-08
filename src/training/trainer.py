"""Script with the default training loop."""

from pathlib import Path
from typing import Any

import pytorch_lightning as pl
import torch
from src.datamodule.loader import get_datamodule, get_datasets, get_metadata
from src.infmax_models.loader import load_model
from src.training.callbacks import get_callbacks
from src.training.loggers import get_loggers
from src.utils.wrapper import get_device
from src.wrapper.hetero import HetergoGNN_WrapperConfig, HeteroGNN_Wrapper


def train(args: dict[str, Any]) -> None:
    """Main training loop with args provided by YAML config.."""
    datasets = get_datasets(args)
    datamodule = get_datamodule(datasets=datasets, config=args)
    wrapper = HeteroGNN_Wrapper(
        model=load_model(config=args),
        config=HetergoGNN_WrapperConfig(
            loss_name=args["training"]["loss"]["name"],
            loss_args=args["training"]["loss"]["args"],
            optimizer_name=args["training"]["optimizer"]["name"],
            optimizer_args=args["training"]["optimizer"]["args"],
            scheduler_name=scheduler.get("name")
            if (scheduler := args["training"].get("scheduler"))
            else None,
            scheduler_args=scheduler.get("args")
            if (scheduler := args["training"].get("scheduler"))
            else None,
            scheduler_config=scheduler.get("config")
            if (scheduler := args["training"].get("scheduler"))
            else None,
            aggr=args["model"].get("aggr"),
            metadata=get_metadata(list(datasets.values())),
            num_neighbors=args["data"]["num_neighbors"],
            neighbor_batch_size=args["data"]["neighbor_batch_size"],
            device=get_device(args["training"]["devices"])
            if "devices" in args["training"] is not None
            else "cuda"
            if torch.cuda.is_available()
            else "cpu",
        ),
    )
    trainer = pl.Trainer(
        max_epochs=args["training"]["max_epochs"],
        accelerator="gpu" if torch.cuda.is_available() else "cpu",
        devices=args["training"]["devices"]
        if "devices" in args["training"] is not None
        else "auto",
        log_every_n_steps=1,
        callbacks=get_callbacks(args),
        logger=get_loggers(config=args, model=wrapper),
    )
    trainer.fit(model=wrapper, datamodule=datamodule)
    test_output = trainer.test(model=wrapper, datamodule=datamodule)
    wrapper.save_test_result(
        save_path=Path(args["hydra"]["run"]["dir"]), test_output=test_output
    )
