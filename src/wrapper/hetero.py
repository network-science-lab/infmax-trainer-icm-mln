import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import pytorch_lightning as pl
import torch
from _data_set.nsl_data_utils.loaders.constants import ACTOR
from src.infmax_models.base.base import BaseHeteroModule
from src.utils.wrapper import get_loss, get_optimizer, get_scheduler
from torch.optim import Optimizer
from torch_geometric.data.batch import Batch
from torch_geometric.loader.neighbor_loader import NeighborLoader
from torch_geometric.nn import to_hetero_with_bases


@dataclass
class HetergoGNN_WrapperConfig:
    loss_name: str
    loss_args: dict[str, Any]
    optimizer_name: str
    optimizer_args: dict[str, Any]
    scheduler_name: str | None
    scheduler_args: dict[str, Any] | None
    scheduler_config: dict[str, Any] | None
    aggr: str | None
    metadata: tuple
    num_neighbors: list[int]
    neighbours_batch_size: int
    device: str


class HeteroGNN_Wrapper(pl.LightningModule):
    def __init__(
        self,
        model: BaseHeteroModule,
        config: HetergoGNN_WrapperConfig,
    ) -> None:
        super().__init__()
        self._config = config
        self.student = model
        self.is_hetero = model.is_hetero
        if not model.is_hetero:
            self.student = to_hetero_with_bases(
                module=self.student,
                metadata=self._config.metadata,
                num_bases=5, #TODO: Include in config if we choose this aproach
            )
        self._loss = get_loss(
            loss_name=config.loss_name,
            loss_args=config.loss_args,
        )

        self.test_preds = {
            "trues": [],
            "preds": [],
        }
        self.save_hyperparameters(ignore=["model"])

    def _mask_batch(
        self,
        x_dict: dict[str, torch.Tensor],
        edge_index_dict: dict[str, torch.Tensor],
    ) -> tuple[dict[str, torch.Tensor], dict[str, torch.Tensor]]:
        expected_node_types, expected_edge_types = self._config.metadata

        missing_node_types = [
            node_type for node_type in expected_node_types if node_type not in x_dict
        ]

        for node_type in missing_node_types:
            x_dict[node_type] = torch.empty(
                size=0,
            ).to(self._config.device)

        present_edge_types = edge_index_dict.keys()
        missing_edge_types = [
            edge_type
            for edge_type in expected_edge_types
            if edge_type not in present_edge_types
        ]

        for edge_type in missing_edge_types:
            edge_index_dict[edge_type] = torch.empty(
                size=[edge_index_dict[list(present_edge_types)[0]].shape[0], 0],
                dtype=torch.long,
            ).to(self._config.device)

        return (
            x_dict,
            edge_index_dict,
        )

    def forward(
        self,
        x_dict: dict[str, torch.Tensor],
        z_dict: dict[str, torch.Tensor],
        edge_index_dict: dict[str, torch.Tensor],
    ) -> dict[str, torch.Tensor]:
        if not self.is_hetero:
            x_dict, edge_index_dict = self._mask_batch(
                x_dict=x_dict, edge_index_dict=edge_index_dict
            )
        return self.student.forward(x_dict, z_dict, edge_index_dict)

    def _calculate_loss(
        self,
        batch: Batch,
        predictions: dict[str, torch.Tensor],
    ) -> torch.Tensor:
        return self._loss(
            predictions[ACTOR][:batch[ACTOR].batch_size],
            batch[ACTOR].y[:batch[ACTOR].batch_size],
        )

    def _get_neighbour_loader(
        self,
        batch: Batch,
    ) -> NeighborLoader:
        return NeighborLoader(
            data=batch,
            num_neighbors={
                relation: self._config.num_neighbors
                for relation in self._config.metadata[1]
            },
            input_nodes=(ACTOR, None),
            batch_size=self._config.neighbours_batch_size,
            shuffle=True,
            subgraph_type="bidirectional",
            directed=False,
        )

    def training_step(
        self,
        batch: Batch,
        batch_idx: int,
    ) -> torch.Tensor:
        neighbours_loader = self._get_neighbour_loader(batch)

        loss = 0
        for subgraf_batch in neighbours_loader:
            predictions = self.forward(
                x_dict=subgraf_batch.x_dict,
                z_dict=subgraf_batch.z_dict,
                edge_index_dict=subgraf_batch.edge_index_dict,
            )
            loss += self._calculate_loss(
                batch=subgraf_batch,
                predictions=predictions,
            )

        self.log(
            name="train_loss",
            value=loss,
            batch_size=len(batch),
        )

        return loss

    def validation_step(
        self,
        batch: Batch,
        batch_idx: int,
    ) -> torch.Tensor:
        neighbours_loader = self._get_neighbour_loader(batch)

        loss = 0
        with torch.no_grad():
            for subgraf_batch in neighbours_loader:
                predictions = self.forward(
                    x_dict=subgraf_batch.x_dict,
                    z_dict=subgraf_batch.z_dict,
                    edge_index_dict=subgraf_batch.edge_index_dict,
                )
                loss += self._calculate_loss(
                    batch=subgraf_batch,
                    predictions=predictions,
                )

        self.log(
            name="val_loss",
            value=loss,
            batch_size=len(batch),
            prog_bar=True,
            on_epoch=True,
        )

        return loss

    def test_step(
        self,
        batch: Batch,
        batch_idx: int,
    ) -> torch.Tensor:
        neighbours_loader = self._get_neighbour_loader(batch)
        layers = batch.x_dict.keys()

        loss = 0
        with torch.no_grad():
            for subgraf_batch in neighbours_loader:
                predictions = self.forward(
                    x_dict=subgraf_batch.x_dict,
                    z_dict=subgraf_batch.z_dict,
                    edge_index_dict=subgraf_batch.edge_index_dict,
                )
                loss += self._calculate_loss(
                    batch=subgraf_batch,
                    predictions=predictions,
                )

                for layer in layers:
                    self.test_preds["trues"] += batch[layer].y.tolist()
                    self.test_preds["preds"] += predictions[layer].tolist()

        self.log(
            name=f"test_loss_{batch.network_name[0]}",
            value=loss,
            batch_size=len(batch),
        )

        return loss

    def configure_optimizers(self) -> dict[str, Optimizer | dict[str, Any]]:
        configures_optimizers = {
            "optimizer": get_optimizer(
                optimizer_name=self._config.optimizer_name,
                optimizer_args=self._config.optimizer_args,
                model_parameters=self.parameters(),
            ),
        }
        if self._config.scheduler_name:
            configures_optimizers["scheduler"] = get_scheduler(
                scheduler_name=self._config.scheduler_name,
                scheduler_args=self._config.scheduler_args,
                scheduler_config=self._config.scheduler_config,
                optimizer=configures_optimizers["optimizer"],
            )

        return configures_optimizers

    def clear_test_results(self) -> None:
        self.test_preds = {
            "trues": [],
            "preds": [],
        }

    def save_test_result(
        self,
        save_path: Path,
        test_output: list[dict[str, float]],
    ) -> None:
        save_path = save_path / "results_test"
        save_path.mkdir(
            exist_ok=True,
            parents=True,
        )

        with (save_path / f"predictions.json").open(
            mode="w",
            encoding="utf-8",
        ) as file:
            json.dump(
                obj=self.test_preds,
                fp=file,
                indent=2,
            )

        with (save_path / f"metrics.json").open(
            mode="w",
            encoding="utf-8",
        ) as file:
            json.dump(
                obj=test_output,
                fp=file,
                indent=2,
            )
