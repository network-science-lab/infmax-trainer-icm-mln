import logging
from pathlib import Path

import pandas as pd
import torch
from bidict import bidict
from neptune import Run
from torch.optim import Optimizer
from torch.utils.data import DataLoader
from torch_geometric.data import Dataset
from torch_geometric.loader import DataLoader, NeighborLoader
from tqdm import tqdm

from src.data_models.mln_hetero_data import MLNHeteroData


class NeighbourhoodLoaderWrapper:

    def __init__(self, num_workers: int, batch_size: int, batch_neighbours: list[int], batch_subraph_type: str):
        self.num_workers = num_workers
        self.batch_size = batch_size
        self.batch_neighbours = batch_neighbours
        self.batch_subraph_type = batch_subraph_type

    @staticmethod
    def get_neighbour_loader(
        graph_sample: MLNHeteroData, num_workers: int, batch_size: int, batch_neighbours: list[int], batch_subraph_type: str
    ) -> NeighborLoader:
        return NeighborLoader(
            data=graph_sample,
            num_neighbors={key: batch_neighbours for key in graph_sample.edge_types},
            input_nodes=[*graph_sample.node_types, None],
            batch_size=batch_size, 
            shuffle=True,
            subgraph_type=batch_subraph_type,
            num_workers=num_workers,
            pin_memory=True,
        )
    
    def __call__(self, graph_sample: MLNHeteroData) -> NeighborLoader:
        return self.get_neighbour_loader(
            graph_sample=graph_sample,
            num_workers=self.num_workers,
            batch_size=self.batch_size,
            batch_neighbours=self.batch_neighbours,
            batch_subraph_type=self.batch_subraph_type
        )


class BareTorchWrapper:

    @staticmethod
    def train_step(
        model: torch.nn.Module,
        data_set: Dataset,
        data_loader: DataLoader,
        optimizer: Optimizer,
        device: str,
        loss_func: torch.nn.Module,
        logger: Run
    ) -> float:
        model.train()
        epoch_loss = []
        for _, graph in enumerate(tqdm(data_set)):
            graph_loss = []
            for batch in data_loader(graph):
                batch = batch.to(device)
                optimizer.zero_grad()
                out = model.forward(
                    x_dict=batch.x_dict,
                    z_dict=batch.z_dict,
                    edge_index_dict=batch.edge_index_dict,
                )
                loss = loss_func(batch["actor"].y, out["actor"])
                logger["train/batch"].append(loss.item())
                loss.backward()
                optimizer.step()
                graph_loss.append(loss.item())
            logger["train/graph"].append(sum(graph_loss) / len(graph_loss))
            epoch_loss.extend(graph_loss)
        return sum(epoch_loss) / len(epoch_loss)

    @staticmethod
    @torch.no_grad
    def validate_step(
        model: torch.nn.Module,
        data_set: Dataset,
        data_loader: DataLoader,
        device: str,
        loss_func: torch.nn.Module,
        logger: Run
    ) -> float:
        model.eval()
        val_loss = []
        for _, graph in enumerate(tqdm(data_set)):
            graph_loss = []
            for batch in data_loader(graph):
                batch = batch.to(device)
                out = model.forward(
                    x_dict=batch.x_dict,
                    z_dict=batch.z_dict,
                    edge_index_dict=batch.edge_index_dict,
                )
                loss = loss_func(batch["actor"].y, out["actor"])
                graph_loss.append(loss)
            logger["val/graph"].append(sum(graph_loss) / len(graph_loss))
            val_loss.extend(graph_loss)
        return sum(val_loss) / len(val_loss)

    @staticmethod
    def transform_labels(graph: MLNHeteroData, preds: torch.Tensor) -> pd.DataFrame:
        actors_map = bidict({a_id: int(a_idx) for a_id, a_idx in graph.actors_map.items()})
        real_labels = [actors_map.inverse[i] for i in range(preds.shape[0])]
        preds_np = preds.cpu().numpy() * len(graph.actors_map)       
        return pd.DataFrame(preds_np, index=real_labels, columns=graph.y_names).sort_index()

    @torch.no_grad
    def test_step(
        self,
        model: torch.nn.Module,
        graph: MLNHeteroData,
        data_loader: DataLoader,
        device: str,
        loss_func: torch.nn.Module, 
        logger: Run,
        out_dir: Path,
    ) -> list[float]:
        model.eval()
        graph_loss = []
        graph_name = f"{graph.network_type}_{graph.network_name}"
        logging.info(f"Testing: {graph_name}")
        for idx, batch in enumerate(tqdm(data_loader(graph))):
            batch = batch.to(device)
            predictions = model.forward(
                x_dict=batch.x_dict,
                z_dict=batch.z_dict,
                edge_index_dict=batch.edge_index_dict,
            )
            batch_loss = loss_func(batch["actor"].y, predictions["actor"])
            logger[f"test/{graph_name}"].append(batch_loss)
            graph_loss.append(batch_loss)
            # self.transform_labels(graph, graph["actor"].y).to_csv(out_dir / f"{graph_name}_{idx}_y.csv")
            # self.transform_labels(graph, predictions["actor"]).to_csv(out_dir / f"{graph_name}_{idx}_yhat.csv")
        logger[f"test/{graph_name}_avg"].append(sum(graph_loss) / len(graph_loss))
        return graph_loss

    def train_model(
        self,
        model: torch.nn.Module,
        train_dataset: Dataset,
        val_dataset: Dataset,
        data_loader: DataLoader,
        loss_func: torch.nn.Module,
        optimizer: torch.nn.Module,
        num_epochs: int,
        device: str,
        logger: Run,
        out_dir: str
    ) -> None:
        logging.info("Training the model.")
        model.to(device)
        out_dir = Path(out_dir)
        out_dir.mkdir(exist_ok=True, parents=True)
        best_val_loss = float('inf')
        for epoch in range(num_epochs):
            logging.info(f"Epoch {epoch+1}/{num_epochs}")
            train_loss = self.train_step(model, train_dataset, data_loader, optimizer, device, loss_func, logger)
            logger["train/epoch"].append(train_loss)
            val_loss = self.validate_step(model, val_dataset, data_loader, device, loss_func, logger)
            logging.info(f"Epoch {epoch+1}/{num_epochs}: Train Loss = {train_loss:.4f}, Val Loss = {val_loss:.4f}")
            logger["val/epoch"].append(val_loss)
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                torch.save(model.state_dict(), str(out_dir / "best_model.pth"))
                logging.info("Model saved!")

    @torch.no_grad
    def test_model(
        self,
        model: torch.nn.Module,
        test_dataset: Dataset,
        data_loader: DataLoader,
        loss_func: torch.nn.Module,
        device: str,
        logger: Run,
        out_dir: str,
    ) -> None:
        logging.info("Testing the model.")
        model.eval()
        out_dir = Path(out_dir)
        out_dir.mkdir(exist_ok=True, parents=True)
        test_loss = []
        for graph in test_dataset:
            graph_loss = self.test_step(model, graph, data_loader, device, loss_func, logger, out_dir)
            test_loss.extend(graph_loss)
        logger["test/avg"].append(sum(test_loss) / len(test_loss))
