import logging
import torch
from torch.utils.data import DataLoader

from pathlib import Path
from typing import Any

import pytorch_lightning as pl
from lightning.pytorch.loggers import NeptuneLogger
from tqdm import tqdm
from src.data_loader import get_datamodule, get_datasets
from src.data_models.mln_hetero_data import MLNHeteroData
from src.infmax_models.loader import load_model
from src.training.callbacks import get_callbacks
from src.training.loggers import get_loggers, init_neptune
from src.utils.config import validate_config
from src.utils.misc import general_test_result
from src.utils.worker import get_num_workers
from src.utils.wrapper import get_accelerator, get_loss, get_optimizer
from src.wrapper.hetero import HetergoGNNWrapperConfig, HeteroGNNWrapper

from torch_geometric.loader import DataLoader
from neptune.utils import stringify_unsupported


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
    train_loader = DataLoader(datasets["train"], batch_size=1, shuffle=True, num_workers=15)
    val_loader = DataLoader(datasets["val"], batch_size=1, shuffle=True, num_workers=15)
    test_loader = DataLoader(datasets["test"], batch_size=1, shuffle=True, num_workers=15)

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
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    train_model(
        model,
        train_loader,
        val_loader,
        loss_func,
        optimizer,
        num_epochs=args["training"]["max_epochs"],
        device=device,
        logger=logger,
        out_dir="./dupa",
    )
    test_model(
        model=model,
        test_loader=test_loader,
        device=device,
        loss_func=loss_func,
        logger=logger,
        out_dir="./dupa",
    )
    logger.stop()


def train_one_epoch(model, loader, optimizer, device, loss_func, logger):
    model.train()
    epoch_loss = 0.0

    for idx, batch in enumerate(tqdm(loader)):
        batch = batch.to(device)
        optimizer.zero_grad()

        # Forward pass
        out = model.forward(
            x_dict=batch.x_dict,
            z_dict=batch.z_dict,
            edge_index_dict=batch.edge_index_dict,
        )
        loss = loss_func(batch["actor"].y, out["actor"])
        logger["train/batch"].append(loss.item())

        # Backward pass
        loss.backward()
        optimizer.step()

        epoch_loss += loss.item()

    avg_loss = epoch_loss / len(loader)
    return avg_loss


def validate(model, loader, device, loss_func):
    model.eval()
    val_loss = 0.0
    with torch.no_grad():
        for batch in loader:
            batch = batch.to(device)
            out = model.forward(
                x_dict=batch.x_dict,
                z_dict=batch.z_dict,
                edge_index_dict=batch.edge_index_dict,
            )
            loss = loss_func(batch["actor"].y, out["actor"])
            val_loss += loss

    avg_val_loss = val_loss / len(loader)
    return avg_val_loss


def train_model(model, train_loader, val_loader, loss_func, optimizer, num_epochs, device, logger, out_dir: str):
    model.to(device)
    out_dir = Path(out_dir)
    out_dir.mkdir(exist_ok=True, parents=True)
    best_val_loss = float('inf')
    for epoch in range(num_epochs):
        logging.info(f"epoch {epoch}/{num_epochs}")
        train_loss = train_one_epoch(model, train_loader, optimizer, device, loss_func, logger)
        logger["train/epoch"].append(train_loss)
        val_loss = validate(model, val_loader, device, loss_func)
        logging.info(f"Epoch {epoch+1}/{num_epochs}: Train Loss = {train_loss:.4f}, Val Loss = {val_loss:.4f}")
        logger["val/epoch"].append(val_loss)
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save(model.state_dict(), str(out_dir / "best_model.pth"))
            logging.info("Model saved!")



#### test


from pathlib import Path
import pandas as pd
from bidict import bidict


@torch.no_grad
def test_step(model, graph: MLNHeteroData, loss_func, logger, out_dir) -> torch.Tensor:
    graph_name = f"{graph.network_type[0]}_{graph.network_name[0]}"
    predictions = model.forward(
        x_dict=graph.x_dict,
        z_dict=graph.z_dict,
        edge_index_dict=graph.edge_index_dict,
    )
    loss = loss_func(graph["actor"].y, predictions["actor"])
    logger[f"test/{graph_name}"].append(loss)
    transform_labels(graph, graph["actor"].y).to_csv(out_dir / f"{graph_name}_y.csv")
    transform_labels(graph, predictions["actor"]).to_csv(out_dir / f"{graph_name}_yhat.csv")
    return loss


def transform_labels(graph: MLNHeteroData, preds: torch.Tensor) -> pd.DataFrame:
    actors_map = bidict({a_id: int(a_idx) for a_id, a_idx in graph.actors_map.items()})
    real_labels = [actors_map.inverse[i] for i in range(preds.shape[0])]
    preds_np = preds.cpu().numpy() * len(graph.actors_map)       
    return pd.DataFrame(preds_np, index=real_labels, columns=graph.y_names).sort_index()


def test_model(model, test_loader, device, loss_func, logger, out_dir: str):
    logging.info("Testing the model.")
    model.eval()
    out_dir = Path(out_dir)
    out_dir.mkdir(exist_ok=True, parents=True)
    test_loss = 0.0
    for graph in test_loader:
        graph_loss = test_step(model, graph.to(device), loss_func, logger, out_dir)
        test_loss += graph_loss
    avg_test_loss = test_loss / len(test_loader)
    logger["test/avg"].append(avg_test_loss)
    return avg_test_loss
