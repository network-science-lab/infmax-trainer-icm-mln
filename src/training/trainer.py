"""Script with the default training loop."""

from pathlib import Path
from typing import Any

import pytorch_lightning as pl
from src.data_loader import get_datamodule, get_datasets
from src.infmax_models.loader import load_model
from src.training.callbacks import get_callbacks
from src.training.loggers import get_loggers
from src.utils.config import validate_config
from src.utils.misc import general_test_result
from src.utils.worker import get_num_workers
from src.utils.wrapper import get_accelerator
from src.wrapper.hetero import HetergoGNNWrapperConfig, HeteroGNNWrapper


def train(args: dict[str, Any]) -> None:
    """Main training loop with args provided by YAML config.."""
    validate_config(args)
    
    datasets = get_datasets(args)
    # from tqdm import tqdm
    # for sample in tqdm(range(len(datasets["train"]))):
    #     datasets["train"][sample]
    # for sample in tqdm(range(len(datasets["val"]))):
    #     datasets["val"][sample]
    # for sample in tqdm(range(len(datasets["test"]))):
    #     datasets["test"][sample]

    datamodule = get_datamodule(datasets=datasets, config=args)
    scheduler = args["training"].get("scheduler")
    device = args["training"].get("devices")
    wrapper = HeteroGNNWrapper(
        model=load_model(config=args),
        config=HetergoGNNWrapperConfig(
            loss_name=args["training"]["loss"]["name"],
            loss_args=args["training"]["loss"]["args"],
            optimizer_name=args["training"]["optimizer"]["name"],
            optimizer_args=args["training"]["optimizer"]["args"],
            scheduler_name=scheduler.get("name") if scheduler else None,
            scheduler_args=scheduler.get("args") if scheduler else None,
            scheduler_config=scheduler.get("config") if scheduler else None,
            batch_size=args["data"]["batch"]["size"],
            batch_neighbours=args["data"]["batch"]["neighbours_sampling"],
            batch_subraph_type=args["data"]["batch"]["subgraph_type"],
            num_workers=get_num_workers(config=args),
        ),
    )
    logger = get_loggers(config=args, model=wrapper)
    trainer = pl.Trainer(
        max_epochs=args["training"]["max_epochs"],
        accelerator=get_accelerator(args["training"].get("accelerator")),
        devices=device if device else "auto",
        log_every_n_steps=1,
        callbacks=get_callbacks(args),
        logger=logger,
    )
    trainer.fit(model=wrapper, datamodule=datamodule)
    test_output = trainer.test(model=wrapper, datamodule=datamodule)
    test_output.append(general_test_result(test_output))
    logger[0].log_metrics(test_output[-1])
    wrapper.save_test_result(save_path=Path(args["hydra"]["run"]["dir"]), test_output=test_output)
