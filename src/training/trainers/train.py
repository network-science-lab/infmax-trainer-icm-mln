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
from src.netsp_models.mln_info import MultilayerNetworkInfo
from src.training.callbacks import get_callbacks
from src.training.loggers import get_loggers
from src.training.trainers import TRAINABLE
from src.training.trainers.eval import evaluate_seed_set
from src.wrapper.hetero import HetergoGNN_WrapperConfig, HeteroGNN_Wrapper

# TODO: prepare unified pipeline for directly/indirectly trainable method with common interface for invocation and inference
# TODO: probably we need to update indirectly_trainable function

def indirectly_trainable(args: dict[str, Any]) -> None:
    # load dataset
    networks = [
        MultilayerNetworkInfo(
            name=n,
            net_pt=load_network(
                net_name=n,
                as_tensor=True,
            ),
            y_type=None,
            sp_raw=None,
            icm_protocol=args["spreading_regime"]["protocol"],
            x_type=None,
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
        logging.info(f"Dataset: {net.name}")

        pred_seeds = model(network=net.net_pt)
        pred_performance = evaluate_seed_set(
            net=net.net_pt,
            seed_set=pred_seeds,
            protocol=proto,
            probability=p,
            n_steps=n_steps,
            n_repetitions=n_repetitions,
        )
        logging.info(f"Predicted seed set: {pred_seeds}")
        logging.info(f"{pred_performance.mean()}\n")

        ref_seeds = get_gt_data(net.name, proto, p, seed_size)
        ref_performance = evaluate_seed_set(
            net=net.net_pt,
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
    datamodule = get_datamodule(datasets=datasets, config=args)
    wrapper = HeteroGNN_Wrapper(
        model=load_model(config=args),
        config=HetergoGNN_WrapperConfig(
            loss_name=args["training"]["loss"]["name"],
            loss_args=args["training"]["loss"]["args"],
            learning_rate=args["training"]["learning_rate"],
            aggr=args["model"]["aggr"],
            metadata=get_metadata(list(datasets.values())),
            device="cuda" if torch.cuda.is_available() else "cpu",
        ),
    )
    trainer = pl.Trainer(
        max_epochs=args["training"]["max_epochs"],
        accelerator="gpu" if torch.cuda.is_available() else "cpu",
        devices=args["training"]["devices"] if "devices" in args["training"] is not None else "auto",
        log_every_n_steps=1,
        callbacks=get_callbacks(args),
        logger=get_loggers(config=args, model=wrapper),
    )
    trainer.fit(model=wrapper, datamodule=datamodule)
    test_output = trainer.test(model=wrapper, datamodule=datamodule)
    wrapper.save_test_result(save_path=Path(args["hydra"]["run"]["dir"]), test_output=test_output)


def train(args: dict[str, Any]) -> None:
    if args["model"]["name"] in TRAINABLE:
        directly_trainable(args)
    else:
        indirectly_trainable(args)
