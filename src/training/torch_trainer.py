"""Bare torch trainer - only for debugging purposes."""

import logging
from typing import Any, Dict, Union
from unittest.mock import MagicMock

import torch
import neptune.new
import neptune

from neptune.utils import stringify_unsupported
from src.data_module import get_datasets
from src.infmax_models.loader import load_model
from src.training.loggers import init_neptune
from src.utils.config import validate_config
from src.utils.worker import get_num_workers
from src.utils.wrapper import get_loss, get_optimizer
from src.wrapper.bare_torch_wrapper import BareTorchWrapper, NeighbourhoodLoaderWrapper, DataLoader


def init_neptune(config: Dict) -> Union[MagicMock, neptune.Run]:
    try:
        logging.getLogger("neptune").setLevel(logging.CRITICAL)
        neptune_instance = create_neptune_instance(config)
    except Exception as e:
        logging.info(e)
        logging.info("Neptune not initialized - using mocked logger!")
        neptune_instance = MagicMock()
    return neptune_instance


def create_neptune_instance(config: dict[str, Any]):
    neptune_instance = neptune.init_run(
        project="infmax/infmax-gnn",
        api_token=os.getenv(
            key="NEPTUNE_API_KEY",
            default=neptune.ANONYMOUS_API_TOKEN,
        ),
        name=config["model"]["name"],
        tags=[tag_config["name"] for tag_config in config["training"]["logger"]["tags"]],
    )
    return neptune_instance


def log_geometric_model_summary(logger, model):
    layers = []
    for name, module in model.named_children():
        layers.append(f"{name}: {module}")
    model_summary = "\n".join(layers)
    logger["model/summary"] = model_summary
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    logger["model/total_params"] = total_params
    logger["model/trainable_params"] = trainable_params


def train(args: dict[str, Any]) -> None:
    """Main training loop with args provided by YAML config.."""
    validate_config(args)
    logger = init_neptune(config=args)
    logger["hyperparams"] = stringify_unsupported(
        {key: value for key, value in args.items() if key != "hydra"}
    )

    datasets = get_datasets(args)
    neighour_loader = NeighbourhoodLoaderWrapper(
        num_workers=get_num_workers(config=args),
        batch_size=args["data"]["batch"]["size"],
        batch_neighbours=args["data"]["batch"]["neighbours_sampling"],
        batch_subraph_type=args["data"]["batch"]["subgraph_type"],
    )
    # train_loader = DataLoader(datasets["train"], batch_size=1, shuffle=True, num_workers=get_num_workers(config=args))
    # val_loader = DataLoader(datasets["val"], batch_size=1, shuffle=True, num_workers=get_num_workers(config=args))
    # test_loader = DataLoader(datasets["test"], batch_size=1, shuffle=False, num_workers=get_num_workers(config=args))

    # for sample in tqdm(range(len(datasets["test"]))):
    #     graph = datasets["test"][sample]
    #     logging.critical(f"{graph.network_type}_{graph.network_name}")
    #     for subgraph in neighour_loader(graph):
    #         logging.critical(f"{subgraph.network_type}_{subgraph.network_name}")         

    model=load_model(config=args)
    log_geometric_model_summary(logger, model)

    optimizer = get_optimizer(
        optimizer_args=args["training"]["optimizer"]["args"],
        optimizer_name=args["training"]["optimizer"]["name"],
        model_parameters=model.parameters(),
    )
    loss_func = get_loss(
        loss_name=args["training"]["loss"]["name"],
        loss_args=args["training"]["loss"]["args"],
    )

    trainer = BareTorchWrapper()
    trainer.train_model(
        model,
        datasets["train"],
        datasets["val"],
        neighour_loader,
        loss_func,
        optimizer,
        num_epochs=args["training"]["max_epochs"],
        device="cuda" if torch.cuda.is_available() else "cpu",
        logger=logger,
        out_dir="./dupa",
    )
    trainer.test_model(
        model=model,
        test_dataset=datasets["test"],
        data_loader=neighour_loader,
        loss_func=loss_func,
        device="cuda" if torch.cuda.is_available() else "cpu",
        logger=logger,
        out_dir="./dupa",
    )
    logger.stop()
