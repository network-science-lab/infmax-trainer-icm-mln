import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import pytorch_lightning as pl
import torch
from torch_geometric.data.batch import Batch
from torch_geometric.nn import to_hetero

from src.infmax_models.base.base import BaseHeteroModule
from src.utils.wrapper import get_loss


@dataclass
class HetergoGNN_WrapperConfig:
    loss_name: str
    loss_args: dict[str, Any]
    learning_rate: float
    aggr: str
    metadata: tuple
    device: str


# TODO: CONSIDER WIGHTED SUM, WEIGHTS ASSIGNED TO THE LAYERS
class HeteroGNN_Wrapper(pl.LightningModule):
    def __init__(
        self,
        model: BaseHeteroModule,
        config: HetergoGNN_WrapperConfig,
    ) -> None:
        super().__init__()
        self._config = config
        self.student = model
        if not model.is_hetero:
            self.student = to_hetero(
                module=self.student,
                metadata=self._config.metadata,
                aggr=self._config.aggr,
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
        edge_index_dict: dict[str, torch.Tensor],
    ) -> dict[str, torch.Tensor]:
        x_dict, edge_index_dict = self._mask_batch(
            x_dict=x_dict,
            edge_index_dict=edge_index_dict,
        )

        return self.student.forward(x_dict, edge_index_dict)

    def _calculate_loss(
        self,
        batch: Batch,
        predictions: dict[str, torch.Tensor],
    ) -> float:
        loss = 0
        for layer in batch.x_dict.keys():
            loss += self._loss(predictions[layer], batch[layer].y)

        return loss

    def training_step(
        self,
        batch: Batch,
        batch_idx: int,
    ) -> torch.Tensor:
        predictions = self(batch.x_dict, batch.edge_index_dict)
        loss = self._calculate_loss(
            batch=batch,
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
        with torch.no_grad():
            predictions = self(batch.x_dict, batch.edge_index_dict)

        loss = self._calculate_loss(
            batch=batch,
            predictions=predictions,
        )
        self.log(
            name="val_loss",
            value=loss,
            batch_size=len(batch),
        )

        return loss

    def test_step(
        self,
        batch: Batch,
        batch_idx: int,
    ) -> torch.Tensor:
        with torch.no_grad():
            predictions = self(batch.x_dict, batch.edge_index_dict)

        loss = self._calculate_loss(
            batch=batch,
            predictions=predictions,
        )
        self.log(
            name="test_loss",
            value=loss,
            batch_size=len(batch),
        )

        layers = batch.x_dict.keys()
        for layer in layers:
            self.test_preds["trues"] += batch[layer].y.tolist()
            self.test_preds["preds"] += torch.argmax(
                input=predictions[layer],
                dim=1,
            ).tolist()

        return loss

    def configure_optimizers(self) -> torch.optim.Adam:
        optimizer = torch.optim.Adam(
            params=self.parameters(),
            lr=self._config.learning_rate,
        )

        return optimizer

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
