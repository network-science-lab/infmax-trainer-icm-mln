"""
load dataset
load model
train model
evaluate model
"""
import logging
from pathlib import Path
from typing import Any

import pytorch_lightning as pl
import torch

from _data_set.nsl_data_utils.loaders.net_loader import load_network
from _data_set.nsl_data_utils.loaders.sp_loader import get_gt_data
from src.datamodule.loader import get_datamodule, get_datasets, get_metadata
from src.infmax_models.loader import load_model
from src.training.callbacks import get_callbacks
from src.training.loggers import get_loggers
from src.training.trainers import TRAINABLE
from src.training.trainers.eval import evaluate_seed_set
from src.utils.multilayer_network import MultilayerNetworkInfo
from src.wrapper.hetero import HetergoGNN_WrapperConfig, HeteroGNN_Wrapper

# TODO: prepare unified pipeline for directly/indirectly trainable method with common interface for invocation and inference


def indirectly_trainable(args: dict[str, Any]) -> None:
    # load dataset
    networks = [
        MultilayerNetworkInfo(
            network_name=n,
            network=load_network(
                net_name=n,
                as_tensor=True,
            ),
            output_label_name=None,
            spreading_potential=None,
            protocol=args["spreading_regime"]["protocol"],
            features_type=None,
        )
        for n in args["networks"]
    ]

    # load model
    model = load_model(config=args)

    # capture parameters of spreading regime
    proto = args["spreading_regime"]["protocol"]
    p = args["spreading_regime"]["p"]
    n_steps = args["spreading_regime"]["n_steps"]
    n_repetitions = args["spreading_regime"]["n_repetitions"]
    seed_size = args["train"]["seed_size"]

    for net in networks:
        logging.info(f"Dataset: {net.network_name}")

        pred_seeds = model(network=net.network)
        pred_performance = evaluate_seed_set(
            net=net.network,
            seed_set=pred_seeds,
            protocol=proto,
            probability=p,
            n_steps=n_steps,
            n_repetitions=n_repetitions,
        )
        logging.info(f"Predicted seed set: {pred_seeds}")
        logging.info(f"{pred_performance.mean()}\n")

        ref_seeds = get_gt_data(net.network_name, proto, p, seed_size)
        ref_performance = evaluate_seed_set(
            net=net.network,
            seed_set=ref_seeds,
            protocol=proto,
            probability=p,
            n_steps=n_steps,
            n_repetitions=n_repetitions,
        )
        logging.info(f"Reference seed set: {ref_seeds}")
        logging.info(f"{ref_performance.mean()}\n")


def directly_trainable(args: dict[str, Any]) -> None:
    datasets = get_datasets(args)
    datamodule = get_datamodule(
        datasets=datasets,
        config=args,
    )

    wrapper = HeteroGNN_Wrapper(
        model=load_model(config=args),
        config=HetergoGNN_WrapperConfig(
            loss_name=args["training"]["loss"]["name"],
            loss_args=args["training"]["loss"]["args"],
            learning_rate=args["training"]["learning_rate"],
            aggr=args["model"]["aggr"],
            metadata=get_metadata(datasets.values()),
            device="cuda" if torch.cuda.is_available() else "cpu",
        ),
    )

    trainer = pl.Trainer(
        max_epochs=args["training"]["max_epochs"],
        accelerator="gpu" if torch.cuda.is_available() else "cpu",
        log_every_n_steps=1,
        callbacks=get_callbacks(args),
        logger=get_loggers(
            config=args,
            model=wrapper,
        ),
    )
    trainer.fit(
        model=wrapper,
        datamodule=datamodule,
    )
    test_output = trainer.test(
        model=wrapper,
        datamodule=datamodule,
    )
    wrapper.save_test_result(
        save_path=Path(args["hydra"]["run"]["dir"]),
        test_output=test_output,
    )


def train(args: dict[str, Any]) -> None:
    if args["model"]["name"] in TRAINABLE:
        directly_trainable(args)
    else:
        indirectly_trainable(args)
