
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, TypeVar

from bidict import bidict
import pandas as pd
import pytorch_lightning as pl
import torch
from _data_set.nsl_data_utils.loaders.constants import ACTOR
from src.data_models.mln_hetero_data import MLNHeteroData
from src.infmax_models.base.base import BaseHeteroModule
from src.utils.wrapper import get_loss, get_optimizer, get_scheduler
from torch.optim import Optimizer
from torch_geometric.loader.neighbor_loader import NeighborLoader


MLNHeteroDataBatch = TypeVar("T")


@dataclass(frozen=True)
class HetergoGNNWrapperConfig:
    loss_name: str
    loss_args: dict[str, Any]
    optimizer_name: str
    optimizer_args: dict[str, Any]
    scheduler_name: str | None
    scheduler_args: dict[str, Any] | None
    scheduler_config: dict[str, Any] | None
    batch_size: int
    batch_neighbours: list[int]
    batch_subraph_type: str
    num_workers: int

    # @classmethod
    # def from_str(cls, obj: str) -> Self:  # TODO: for removal
    #     match = re.match(r"(\w+)\((.*)\)", obj)
    #     args_str = match.group(2)
    #     args_list = [arg.strip() for arg in re.split(r",\s\b(?=\D)", args_str)]
    #     kwargs = {
    #         arg.split("=")[0]: ast.literal_eval(arg.split("=")[1]) for arg in args_list
    #     }


class HeteroGNNWrapper(pl.LightningModule):
    def __init__(self, model: BaseHeteroModule, config: HetergoGNNWrapperConfig) -> None:
        super().__init__()
        self._config = config
        self.student = model
        self.is_hetero = model.is_hetero
        # TODO: remove it later
        # if not model.is_hetero:
        #     self.student = to_hetero_with_bases(
        #         module=self.student,
        #         metadata=self._config.metadata,
        #         num_bases=model.num_bases,
        #     )
        self._loss = get_loss(loss_name=config.loss_name, loss_args=config.loss_args)
        self.test_preds = {"trues": {},"preds": {}}
        self.save_hyperparameters(ignore=["model"])

    # def _mask_batch(  # TODO: for removal
    #     self,
    #     x_dict: dict[str, torch.Tensor],
    #     edge_index_dict: dict[str, torch.Tensor],
    # ) -> tuple[dict[str, torch.Tensor], dict[str, torch.Tensor]]:
    #     expected_node_types, expected_edge_types = self._config.metadata

    #     missing_node_types = [
    #         node_type for node_type in expected_node_types if node_type not in x_dict
    #     ]

    #     sample_node_type = x_dict[list(x_dict.keys())[0]]
    #     for node_type in missing_node_types:
    #         x_dict[node_type] = torch.empty(
    #             size=0,
    #         ).to(sample_node_type.device)

    #     present_edge_types = edge_index_dict.keys()
    #     missing_edge_types = [
    #         edge_type
    #         for edge_type in expected_edge_types
    #         if edge_type not in present_edge_types
    #     ]

    #     samle_edge_type = edge_index_dict[list(present_edge_types)[0]]
    #     for edge_type in missing_edge_types:
    #         edge_index_dict[edge_type] = torch.empty(
    #             size=[samle_edge_type.shape[0], 0],
    #             dtype=torch.long,
    #         ).to(samle_edge_type.device)

    #     return (
    #         x_dict,
    #         edge_index_dict,
    #     )

    def forward(
        self,
        x_dict: dict[str, torch.Tensor],
        z_dict: dict[str, torch.Tensor],
        edge_index_dict: dict[str, torch.Tensor],
    ) -> dict[str, torch.Tensor]:
        # if not self.is_hetero:  # TODO: for removal
        #     x_dict, edge_index_dict = self._mask_batch(
        #         x_dict=x_dict, edge_index_dict=edge_index_dict
        #     )
        return self.student.forward(x_dict, z_dict, edge_index_dict)

    def _calculate_loss(
        self, batch: MLNHeteroDataBatch,
        predictions: dict[str, torch.Tensor],
    ) -> torch.Tensor:
        return self._loss(predictions[ACTOR], batch[ACTOR].y)

    def _get_neighbour_loader(self, graph_sample: MLNHeteroDataBatch) -> NeighborLoader:
        return NeighborLoader(
            data=graph_sample,
            num_neighbors={key: self._config.batch_neighbours for key in graph_sample.edge_types},
            input_nodes=[*graph_sample.node_types, None],
            batch_size=self._config.batch_size, 
            shuffle=True,
            subgraph_type=self._config.batch_subraph_type,
            num_workers=min(len(graph_sample.actors_map) // 100, self._config.num_workers),
            pin_memory=True,
        )

    def training_step(self, batch: MLNHeteroDataBatch, batch_idx: int) -> torch.Tensor:
        loss = 0
        for graph in [batch] if len(batch) == 1 else batch:  # TODO: consider removing this loop
            neighbours_loader = self._get_neighbour_loader(graph)
            for subgraph in neighbours_loader:
                predictions = self.forward(
                    x_dict=subgraph.x_dict,
                    z_dict=subgraph.z_dict,
                    edge_index_dict=subgraph.edge_index_dict,
                )
                loss += self._calculate_loss(batch=subgraph, predictions=predictions)
            self.log(name="train_loss", value=loss, batch_size=len(batch))
        return loss

    @torch.no_grad
    def validation_step(self, batch: MLNHeteroDataBatch, batch_idx: int) -> torch.Tensor:
        loss = 0
        for graph in [batch] if len(batch) == 1 else batch:  # TODO: consider removing this loop
            neighbours_loader = self._get_neighbour_loader(graph)
            for subgraph in neighbours_loader:
                predictions = self.forward(
                    x_dict=subgraph.x_dict,
                    z_dict=subgraph.z_dict,
                    edge_index_dict=subgraph.edge_index_dict,
                )
                loss += self._calculate_loss(batch=subgraph, predictions=predictions)
            self.log(name="val_loss", value=loss, batch_size=len(batch), prog_bar=True, on_epoch=True)
        return loss

    @torch.no_grad
    def test_step(self, batch: MLNHeteroDataBatch, batch_idx: int) -> torch.Tensor:
        loss = 0
        for graph in [batch] if len(batch) == 1 else batch:  # TODO: consider removing this loop; removed neighbourhood loader by intention
            graph_name = f"{graph.network_type[0]}_{graph.network_name[0]}"
            predictions = self.forward(
                x_dict=graph.x_dict,
                z_dict=graph.z_dict,
                edge_index_dict=graph.edge_index_dict,
            )
            loss += self._calculate_loss(batch=graph, predictions=predictions)
            self.log(name=f"test_loss_{graph_name}", value=loss, batch_size=len(graph))
            self.test_preds["preds"][graph_name] = self.transform_labels(graph, predictions[ACTOR])
            self.test_preds["trues"][graph_name] = self.transform_labels(graph, graph[ACTOR].y)
        return loss

    @staticmethod
    def transform_labels(graph: MLNHeteroData, preds: torch.Tensor) -> pd.DataFrame:
        actors_map = bidict({a_id: int(a_idx) for a_id, a_idx in graph.actors_map.items()})
        real_labels = [actors_map.inverse[i] for i in range(preds.shape[0])]
        preds_np = preds.cpu().numpy() * len(graph.actors_map)       
        return pd.DataFrame(preds_np, index=real_labels, columns=graph.y_names).sort_index()

    # @torch.no_grad
    # def predict_step(  # TODO: refactor this; use code from test step since it's almost similar
    #     self,
    #     batch: MLNHeteroDataBatch,
    #     batch_idx: int,
    # ) -> dict[str, torch.Tensor]:
    #     layers = batch.x_dict.keys()
    #     batch = self._get_neighbour_loader(
    #         graph_sample=batch,
    #         shuffle=False,
    #         subgraph_type="induced",
    #     )
    #     result = {layer: {} for layer in layers}

    #     for subgraf_batch in batch:
    #         predictions = self.forward(
    #             x_dict=subgraf_batch.x_dict,
    #             z_dict=subgraf_batch.z_dict,
    #             edge_index_dict=subgraf_batch.edge_index_dict,
    #         )

    #         for layer in layers:
    #             subgraf_batch_size = subgraf_batch[layer].batch_size
    #             for idx, key in enumerate(
    #                 subgraf_batch[layer].n_id[:subgraf_batch_size]
    #             ):
    #                 result[layer][key.tolist()] = predictions[layer][
    #                     :subgraf_batch_size
    #                 ][idx]

    #     return result

    def configure_optimizers(self) -> dict[str, Optimizer | dict[str, Any]]:
        configures_optimizers = {
            "optimizer": get_optimizer(
                optimizer_name=self._config.optimizer_name,
                optimizer_args=self._config.optimizer_args,
                model_parameters=self.parameters(),
            ),
        }
        if self._config.scheduler_name:
            configures_optimizers["lr_scheduler"] = get_scheduler(
                scheduler_name=self._config.scheduler_name,
                scheduler_args=self._config.scheduler_args,
                scheduler_config=self._config.scheduler_config,
                optimizer=configures_optimizers["optimizer"],
            )

        return configures_optimizers

    def clear_test_results(self) -> None:
        self.test_preds = {"trues": {}, "preds": {}}

    def save_test_result(self, save_path: Path, test_output: list[dict[str, float]]) -> None:
        save_path = save_path / "results_test"
        save_path.mkdir(exist_ok=True, parents=True)
        for test_name, test_pred in self.test_preds["preds"].items():
            test_pred.to_csv(save_path / f"{test_name}_pred.csv")
        for test_name, test_true in self.test_preds["trues"].items():
            test_true.to_csv(save_path / f"{test_name}_true.csv")
        with (save_path / f"metrics.json").open(mode="w", encoding="utf-8") as file:
            json.dump(obj=test_output, fp=file, indent=2)
