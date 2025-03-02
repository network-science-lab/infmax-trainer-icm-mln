import logging
import os
from pathlib import Path
from typing import Any, Callable
import warnings

import hydra
import neptune
import network_diffusion as nd
import numpy as np
import pandas as pd
import torch
from bidict import bidict
from dotenv import load_dotenv
from omegaconf import DictConfig
from tqdm import tqdm

from _data_set.nsl_data_utils.loaders.constants import EXPOSED, PEAK_INFECTED, PEAK_ITERATION, SIMULATION_LENGTH
from _data_set.nsl_data_utils.loaders.net_loader import load_network
from _data_set.nsl_data_utils.loaders.sp_loader import load_sp
from src import CONFIGS_PATH
from src.datamodule.loader import get_dataset, get_transform
from src.data_models.mln_hetero_data import MLNHeteroData
from src.data_models.mln_info import MLNInfo
from src.dataset.super_spreaders_dataset import SuperSpreadersDataset
from src.infmax_models.loader import load_model
from src.wrapper.mln_hetero import HetergoGNNWrapperConfig, HeteroGNNWrapper
from src.utils.config import load_config
from src.utils.misc import set_seed


load_dotenv(
    dotenv_path=Path(__file__).parent / ".env",
    override=True,
)


# def weighted_sum(
#     score: torch.Tensor,
#     weights: torch.Tensor = torch.Tensor([4, 1, 1, 1]),
# ) -> torch.Tensor:
#     return torch.sum(score * weights)


class HeteroGNN_Predictor:

    def __init__(self, config: dict[str, Any]) -> None:

        self._eval_config = config
    
        evaluation_dir = Path(f'_data_results/evaluation/{config["base"]["run_id"]}')
        evaluation_dir.mkdir(exist_ok=True, parents=True)
        self.evaluation_dir = evaluation_dir

        if random_seed := config["base"].get("random_seed"):
            logging.info(f"Setting randomness seed as {random_seed}!")
            set_seed(config["base"]["random_seed"])

        wrapper_obj, wrapper_config = self.from_neptune(config)
        self._wrapper_obj = wrapper_obj
        self._wrapper_obj.eval()
        self._wrapper_config = wrapper_config

        self._inference_func = self._get_inference_func(config["base"]["selection_function"])
    
    def __call__(self, *args, **kwargs) -> pd.DataFrame:
        return self._inference_func(*args, **kwargs)

    def _get_inference_func(self, func_name: str) -> Callable:
        if func_name == "elimination_approach":
            raise AttributeError
            return self.elimination_approach
        if func_name == "elimination_approach_bis":
            raise AttributeError
            return self.elimination_approach_bis
        elif func_name == "top_k_approach":
            return self.top_k_approach
        raise AttributeError(f"Unknown selecton function: {func_name}")

    def from_neptune(self, run_config: dict[str, Any]) -> tuple[HeteroGNNWrapper, dict[str, Any]]:
        run = neptune.init_run(
            api_token=os.getenv(
                key="NEPTUNE_API_KEY",
                default=neptune.ANONYMOUS_API_TOKEN,
            ),
            project=run_config["base"]["project"],
            with_id=run_config["base"]["run_id"],
            mode="read-only",
        )

        temp = run["training/model/best_model_path"].fetch()
        neptune_best_ckpt_path = (f"training/model/checkpoints/{temp.split('/')[-1].split('.')[0]}")
        local_best_ckpt_path = str(self.evaluation_dir / "best.ckpt")
        run[neptune_best_ckpt_path].download(destination=local_best_ckpt_path)

        model_config = {
            "model": {
                "name": run["training/hyperparams/model/name"].fetch(),
                "parameters": run["training/hyperparams/model/parameters"].fetch(),
            }
        }
        model = load_model(model_config)

        wrapper_config = HetergoGNNWrapperConfig.from_str(run["training/hyperparams/config"].fetch())
        wrapper = HeteroGNNWrapper.load_from_checkpoint(
            checkpoint_path=local_best_ckpt_path,
            model=model,
            config=wrapper_config,
        ).to(run_config["base"]["device"])

        run_config = {
            "model": model_config["model"],
            "data": run["training/hyperparams/data"].fetch(),
        }
    
        return wrapper, run_config

    # def elimination_approach(
    #     self,
    #     network_type: str,
    #     network_name: str,
    # ) -> pd.DataFrame:
    #     output_weights = torch.Tensor(
    #         [
    #             self._config["data"]["output_weights"]["w_e"],
    #             self._config["data"]["output_weights"]["w_sl"],
    #             self._config["data"]["output_weights"]["w_pit"],
    #             self._config["data"]["output_weights"]["w_pin"],
    #         ]
    #     ).to(self._config["base"]["device"])

    #     mln_info = MLNInfo.from_config(
    #         mln_type=network_type,
    #         mln_name=network_name,
    #         icm_protocol=self._config["data"]["protocol"],
    #         icm_p=self._config["base"]["icm_p"],
    #         x_type=self._config["data"]["features_type"],
    #         y_type=self._config["data"]["output_label_name"],
    #     )
    #     net = load_network(
    #         net_type=network_type,
    #         net_name=network_name,
    #         as_tensor=False,
    #     )
    #     sp_df = load_sp(mln_info.sp_paths)

    #     top_spreader = None
    #     top_spreaders = []
    #     top_spreader_potentials = []
    #     for _ in range(self._config["base"]["nb_seeds"]):
    #         ts_actor = top_spreader[0] if top_spreader != None else top_spreader
    #         not_ts_actors = [
    #             actor for actor in net.get_actors() if str(actor.actor_id) != ts_actor
    #         ]
    #         net = net.subgraph(not_ts_actors)
    #         sp_df = sp_df[sp_df["actor"] != str(top_spreader)]

    #         network = MLNHeteroData.from_mln_network(
    #             mln_torch=nd.MultilayerNetworkTorch.from_mln(net),
    #             sp_df=sp_df,
    #             network_info=mln_info,
    #             output_dim=self._config["model_config"]["parameters"]["output_dim"],
    #             input_dim=self._config["model_config"]["parameters"]["input_dim"],
    #         ).to(self._config["base"]["device"])

    #         data = self._wrapper.predict_step(
    #             batch=network,
    #             batch_idx=0,
    #         )

    #         weighted_sums = {
    #             actor: weighted_sum(
    #                 score=spreading_potential,
    #                 weights=output_weights,
    #             )
    #             for actor, spreading_potential in data["actor"].items()
    #         }
    #         max_key = max(weighted_sums, key=weighted_sums.get)

    #         top_spreader_potential = data["actor"][max_key].cpu().numpy() * len(
    #             network.actors_map
    #         )
    #         top_spreader_potentials.append(top_spreader_potential)
    #         top_spreader = self.convert_seed_set(
    #             seeds=[max_key],
    #             actors_map=bidict(
    #                 {
    #                     str(actor): actors_map
    #                     for actor, actors_map in network.actors_map.items()
    #                 }
    #             ),
    #         )
    #         top_spreaders.extend(top_spreader)

    #     df = pd.DataFrame(
    #         data=top_spreader_potentials,
    #         index=top_spreaders,
    #         columns=network.y_names,
    #     ).sort_index()
    #     return df
    
    # @staticmethod
    # def compare_tensors(t1: torch.Tensor, t2: torch.Tensor, weights: torch.Tensor) -> int:
    #     for i in reversed(range(len(weights))):
    #         if t1[i] > t2[i]:
    #             return 1
    #         elif t1[i] < t2[i]:
    #             return -1
    #     return 0

    # def best_actor(self, dict_tensory, weights):
    #     _best_actor = None
    #     _best_result = None
    #     for klucz, tensor in dict_tensory.items():
    #         if _best_actor is None or self.compare_tensors(tensor, _best_result, weights) > 0:
    #             _best_actor = klucz
    #             _best_result = tensor
    #     return _best_actor
  
    # def elimination_approach_bis(
    #     self,
    #     network_type: str,
    #     network_name: str,
    # ) -> pd.DataFrame:
    #     output_weights = torch.Tensor(
    #         [
    #             self._config["data"]["output_weights"]["w_e"],
    #             self._config["data"]["output_weights"]["w_sl"],
    #             self._config["data"]["output_weights"]["w_pit"],
    #             self._config["data"]["output_weights"]["w_pin"],
    #         ]
    #     ).to(self._config["base"]["device"])

    #     mln_info = MLNInfo.from_config(
    #         mln_type=network_type,
    #         mln_name=network_name,
    #         icm_protocol=self._config["data"]["protocol"],
    #         icm_p=self._config["base"]["icm_p"],
    #         x_type=self._config["data"]["features_type"],
    #         y_type=self._config["data"]["output_label_name"],
    #     )
    #     net = load_network(
    #         net_type=network_type,
    #         net_name=network_name,
    #         as_tensor=False,
    #     )
    #     sp_df = load_sp(mln_info.sp_paths)

    #     top_spreader = None
    #     top_spreaders = []
    #     top_spreader_potentials = []
    #     for _ in range(self._config["base"]["nb_seeds"]):
    #         ts_actor = top_spreader[0] if top_spreader != None else top_spreader
    #         not_ts_actors = [
    #             actor for actor in net.get_actors() if str(actor.actor_id) != ts_actor
    #         ]
    #         net = net.subgraph(not_ts_actors)
    #         sp_df = sp_df[sp_df["actor"] != str(top_spreader)]

    #         network = MLNHeteroData.from_mln_network(
    #             mln_torch=nd.MultilayerNetworkTorch.from_mln(net),
    #             sp_df=sp_df,
    #             network_info=mln_info,
    #             output_dim=self._config["model_config"]["parameters"]["output_dim"],
    #             input_dim=self._config["model_config"]["parameters"]["input_dim"],
    #         ).to(self._config["base"]["device"])

    #         data = self._wrapper.predict_step(
    #             batch=network,
    #             batch_idx=0,
    #         )

    #         max_key = self.best_actor(data["actor"], output_weights)

    #         top_spreader_potential = data["actor"][max_key].numpy() * len(
    #             network.actors_map
    #         )
    #         top_spreader_potentials.append(top_spreader_potential)
    #         top_spreader = self.convert_seed_set(
    #             seeds=[max_key],
    #             actors_map=bidict(
    #                 {
    #                     str(actor): actors_map
    #                     for actor, actors_map in network.actors_map.items()
    #                 }
    #             ),
    #         )
    #         top_spreaders.extend(top_spreader)

    #     df = pd.DataFrame(
    #         data=top_spreader_potentials,
    #         index=top_spreaders,
    #         columns=network.y_names,
    #     ).sort_index()

    #     return df

    def prepare_dataset(self, network_type: str, network_name: str) -> SuperSpreadersDataset:
        mln_info = MLNInfo.from_config(
            mln_type=network_type,
            mln_name=network_name,
            icm_protocol=self._wrapper_config["data"]["icm"]["protocol"],
            icm_p=self._wrapper_config["data"]["icm"]["p"],
            x_type=self._eval_config["data"]["features_type"],  # TODO - use values from the training
            y_type=self._eval_config["data"]["output_label_name"],  # TODO - use values from the training
        )
        transform = get_transform(self._wrapper_config["data"]["transform"])
        return SuperSpreadersDataset(
            networks = [mln_info],
            input_dim=self._wrapper_config["model"]["parameters"]["input_dim"],
            output_dim=self._wrapper_config["model"]["parameters"]["output_dim"],
            transform=transform,
        )

    def top_k_approach(self, network_type: str, network_name: str) -> pd.DataFrame:
        dataset = self.prepare_dataset(network_type, network_name)
        mln_hetero_data = dataset[0]
        mln_hetero_data.to(self._eval_config["base"]["device"])
        with warnings.catch_warnings(record=True) as _:  # it's due to an attempt to log to neptune
            self._wrapper_obj.test_step(batch=mln_hetero_data, batch_idx=0)
        prediction_raw = self._wrapper_obj.test_preds["preds"][
            f"{mln_hetero_data.network_type[0]}_{mln_hetero_data.network_name[0]}"
        ]  # this is due to the lightning bug
        prediction_sorted = prediction_raw.sort_values(
            [EXPOSED, SIMULATION_LENGTH, PEAK_INFECTED, PEAK_ITERATION],
            ascending=[False, True, True, False]
        )
        # prediction_sorted = prediction_raw.sort_values(
        #     [EXPOSED],
        #     ascending=[False]
        # )
        return prediction_sorted[:self._eval_config["base"]["nb_seeds"]]  # .index.to_list()
  
    # @staticmethod
    # def convert_seed_set(
    #     seeds: list,
    #     actors_map: bidict,
    # ) -> list[str]:
    #     """Convert indices of the actors_map into real names of actors from the network."""
    #     return [actors_map.inverse[seed] for seed in seeds]


@hydra.main(
    version_base=None,
    config_path=str(CONFIGS_PATH),
    config_name="hydra",
)
def main(cfg: DictConfig) -> None:
    config = load_config(
        cfg=cfg,
        cofig_path=CONFIGS_PATH / "evaluation.yaml",
    )
    logging.info(f"Loaded config: {config}")

    for run_id in config["base"]['run_ids']:
        logging.info(f"Run: {run_id}")

        config["base"]['run_id'] = run_id['id']
        config["data"]['features_type'] = run_id['features_type'] # TODO: remove
        config["data"]['protocol'] = run_id['protocol']  # TODO: remove
        
        evaluator = HeteroGNN_Predictor(config)
        result_path = evaluator.evaluation_dir / "results"
        result_path.mkdir(exist_ok=True, parents=True)

        for network in tqdm(config["run"]["networks"]):
            tqdm.write(f'Processing: {network["name"]}')
            network_ts = evaluator(network_name=network["name"], network_type=network["type"])
            network_ts.to_csv(result_path / f"{network['type']}_{network['name']}.csv")
        logging.info("Evaluation completed")


if __name__ == "__main__":
    main()
