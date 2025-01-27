import logging
import os
from pathlib import Path
from typing import Any

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

from _data_set.nsl_data_utils.loaders.net_loader import load_network
from _data_set.nsl_data_utils.loaders.sp_loader import load_sp
from src import CONFIGS_PATH
from src.data_models.mln_hetero_data import MLNHeteroData
from src.data_models.mln_info import MLNInfo
from src.infmax_models.loader import load_model
from src.utils.config import load_config
from src.utils.misc import set_seed
from src.wrapper.mln_hetero import HetergoGNNWrapperConfig, HeteroGNNWrapper

load_dotenv(
    dotenv_path=Path(__file__).parent / ".env",
    override=True,
)


def weighted_sum(
    score: torch.Tensor,
    weights: torch.Tensor = torch.Tensor([4, 1, 1, 1]),
) -> torch.Tensor:
    return torch.sum(score * weights)


class HeteroGNN_Predictor:
    def __init__(
        self,
        config: dict[str, Any],
    ) -> None:
        evaluation_dir = Path(f'_data_results/evaluation/{config["base"]["run_id"]}')
        evaluation_dir.mkdir(
            exist_ok=True,
            parents=True,
        )
        config["evaluation_dir"] = str(evaluation_dir)

        if random_seed := config["base"].get("random_seed"):
            logging.info(f"Setting randomness seed as {random_seed}!")
            set_seed(config["base"]["random_seed"])

        self._wrapper = self.from_neptune(config)
        self._wrapper.eval()
        self._config = config

    def __call__(
        self,
        network_type: str,
        network_name: str,
        **kwargs,
    ) -> pd.DataFrame:
        if self._config["base"]["selection_function"] == "elimination_approach":
            return self.elimination_approach(
                network_name=network_name,
                network_type=network_type,
            )
        if self._config["base"]["selection_function"] == "elimination_approach_bis":
            return self.elimination_approach_bis(
                network_name=network_name,
                network_type=network_type,
            )
        elif self._config["base"]["selection_function"] == "top_k_approach":
            return self.top_k_approach(
                network_name=network_name,
                network_type=network_type,
            )
        raise AttributeError(
                f"Unknown selecton function: {self._config['base']['selection_function']}"
            )

    @staticmethod
    def from_neptune(config: dict[str, Any]) -> HeteroGNNWrapper:
        run = neptune.init_run(
            api_token=os.getenv(
                key="NEPTUNE_API_KEY",
                default=neptune.ANONYMOUS_API_TOKEN,
            ),
            project=config["base"]["project"],
            with_id=config["base"]["run_id"],
            mode="read-only",
        )

        temp = run["training/model/best_model_path"].fetch()
        neptune_best_ckpt_path = (
            f"training/model/checkpoints/{temp.split('/')[-1].split('.')[0]}"
        )
        local_best_ckpt_path = f"{config['evaluation_dir']}/best.ckpt"
        run[neptune_best_ckpt_path].download(destination=local_best_ckpt_path)

        model_config = {
            "model": {
                "name": run["training/hyperparams/model/name"].fetch(),
                "parameters": run["training/hyperparams/model/parameters"].fetch(),
            }
        }
        model = load_model(model_config)

        wrapper_config = HetergoGNNWrapperConfig.from_str(
            run["training/hyperparams/config"].fetch()
        )
        wrapper = HeteroGNNWrapper.load_from_checkpoint(
            checkpoint_path=local_best_ckpt_path,
            model=model,
            config=wrapper_config,
        ).to(config["base"]["device"])

        config["model_config"] = model_config["model"]

        return wrapper

    def elimination_approach(
        self,
        network_type: str,
        network_name: str,
    ) -> pd.DataFrame:
        output_weights = torch.Tensor(
            [
                self._config["data"]["output_weights"]["w_e"],
                self._config["data"]["output_weights"]["w_sl"],
                self._config["data"]["output_weights"]["w_pit"],
                self._config["data"]["output_weights"]["w_pin"],
            ]
        ).to(self._config["base"]["device"])

        mln_info = MLNInfo.from_config(
            mln_type=network_type,
            mln_name=network_name,
            icm_protocol=self._config["data"]["protocol"],
            icm_p=self._config["base"]["icm_p"],
            x_type=self._config["data"]["features_type"],
            y_type=self._config["data"]["output_label_name"],
        )
        net = load_network(
            net_type=network_type,
            net_name=network_name,
            as_tensor=False,
        )
        sp_df = load_sp(mln_info.sp_paths)

        top_spreader = None
        top_spreaders = []
        top_spreader_potentials = []
        for _ in range(self._config["base"]["nb_seeds"]):
            ts_actor = top_spreader[0] if top_spreader != None else top_spreader
            not_ts_actors = [
                actor for actor in net.get_actors() if str(actor.actor_id) != ts_actor
            ]
            net = net.subgraph(not_ts_actors)
            sp_df = sp_df[sp_df["actor"] != str(top_spreader)]

            network = MLNHeteroData.from_mln_network(
                mln_torch=nd.MultilayerNetworkTorch.from_mln(net),
                sp_df=sp_df,
                network_info=mln_info,
                output_dim=self._config["model_config"]["parameters"]["output_dim"],
                input_dim=self._config["model_config"]["parameters"]["input_dim"],
            ).to(self._config["base"]["device"])

            data = self._wrapper.predict_step(
                batch=network,
                batch_idx=0,
            )

            weighted_sums = {
                actor: weighted_sum(
                    score=spreading_potential,
                    weights=output_weights,
                )
                for actor, spreading_potential in data["actor"].items()
            }
            max_key = max(weighted_sums, key=weighted_sums.get)

            top_spreader_potential = data["actor"][max_key].cpu().numpy() * len(
                network.actors_map
            )
            top_spreader_potentials.append(top_spreader_potential)
            top_spreader = self.convert_seed_set(
                seeds=[max_key],
                actors_map=bidict(
                    {
                        str(actor): actors_map
                        for actor, actors_map in network.actors_map.items()
                    }
                ),
            )
            top_spreaders.extend(top_spreader)

        df = pd.DataFrame(
            data=top_spreader_potentials,
            index=top_spreaders,
            columns=network.y_names,
        ).sort_index()
        return df
    
    @staticmethod
    def compare_tensors(t1: torch.Tensor, t2: torch.Tensor, weights: torch.Tensor) -> int:
        for i in reversed(range(len(weights))):
            if t1[i] > t2[i]:
                return 1
            elif t1[i] < t2[i]:
                return -1
        return 0


    def best_actor(self, dict_tensory, weights):
        _best_actor = None
        _best_result = None
        for klucz, tensor in dict_tensory.items():
            if _best_actor is None or self.compare_tensors(tensor, _best_result, weights) > 0:
                _best_actor = klucz
                _best_result = tensor
        return _best_actor
    
    def elimination_approach_bis(
        self,
        network_type: str,
        network_name: str,
    ) -> pd.DataFrame:
        output_weights = torch.Tensor(
            [
                self._config["data"]["output_weights"]["w_e"],
                self._config["data"]["output_weights"]["w_sl"],
                self._config["data"]["output_weights"]["w_pit"],
                self._config["data"]["output_weights"]["w_pin"],
            ]
        ).to(self._config["base"]["device"])

        mln_info = MLNInfo.from_config(
            mln_type=network_type,
            mln_name=network_name,
            icm_protocol=self._config["data"]["protocol"],
            icm_p=self._config["base"]["icm_p"],
            x_type=self._config["data"]["features_type"],
            y_type=self._config["data"]["output_label_name"],
        )
        net = load_network(
            net_type=network_type,
            net_name=network_name,
            as_tensor=False,
        )
        sp_df = load_sp(mln_info.sp_paths)

        top_spreader = None
        top_spreaders = []
        top_spreader_potentials = []
        for _ in range(self._config["base"]["nb_seeds"]):
            ts_actor = top_spreader[0] if top_spreader != None else top_spreader
            not_ts_actors = [
                actor for actor in net.get_actors() if str(actor.actor_id) != ts_actor
            ]
            net = net.subgraph(not_ts_actors)
            sp_df = sp_df[sp_df["actor"] != str(top_spreader)]

            network = MLNHeteroData.from_mln_network(
                mln_torch=nd.MultilayerNetworkTorch.from_mln(net),
                sp_df=sp_df,
                network_info=mln_info,
                output_dim=self._config["model_config"]["parameters"]["output_dim"],
                input_dim=self._config["model_config"]["parameters"]["input_dim"],
            ).to(self._config["base"]["device"])

            data = self._wrapper.predict_step(
                batch=network,
                batch_idx=0,
            )

            max_key = self.best_actor(data["actor"], output_weights)

            top_spreader_potential = data["actor"][max_key].cpu().numpy() * len(
                network.actors_map
            )
            top_spreader_potentials.append(top_spreader_potential)
            top_spreader = self.convert_seed_set(
                seeds=[max_key],
                actors_map=bidict(
                    {
                        str(actor): actors_map
                        for actor, actors_map in network.actors_map.items()
                    }
                ),
            )
            top_spreaders.extend(top_spreader)

        df = pd.DataFrame(
            data=top_spreader_potentials,
            index=top_spreaders,
            columns=network.y_names,
        ).sort_index()

        return df

    def top_k_approach(
        self,
        network_type: str,
        network_name: str,
    ) -> pd.DataFrame:
        output_weights = torch.Tensor(
            [
                self._config["data"]["output_weights"]["w_e"],
                self._config["data"]["output_weights"]["w_sl"],
                self._config["data"]["output_weights"]["w_pit"],
                self._config["data"]["output_weights"]["w_pin"],
            ]
        ).to(self._config["base"]["device"])

        mln_info = MLNInfo.from_config(
            mln_type=network_type,
            mln_name=network_name,
            icm_protocol=self._config["data"]["protocol"],
            icm_p=self._config["base"]["icm_p"],
            x_type=self._config["data"]["features_type"],
            y_type=self._config["data"]["output_label_name"],
        )
        network = MLNHeteroData.from_network_info(
            network_info=mln_info,
            output_dim=self._config["model_config"]["parameters"]["output_dim"],
            input_dim=self._config["model_config"]["parameters"]["input_dim"],
        ).to(self._config["base"]["device"])

        data = self._wrapper.predict_step(
            batch=network,
            batch_idx=0,
        )

        weighted_sums = {
            actor: weighted_sum(
                score=spreading_potential,
                weights=output_weights,
            )
            for actor, spreading_potential in data["actor"].items()
        }
        sorted_actors = sorted(
            weighted_sums,
            key=weighted_sums.get,
            reverse=True,
        )
        sorted_actors = sorted_actors[: self._config["base"]["nb_seeds"]]
        top_spreader_potentials = np.array(
            [data["actor"][sorted_actor].tolist() for sorted_actor in sorted_actors]
        ) * len(network.actors_map)
        top_spreaders = self.convert_seed_set(
            seeds=sorted_actors,
            actors_map=bidict(
                {
                    str(actor): actors_map
                    for actor, actors_map in network.actors_map.items()
                }
            ),
        )

        df = pd.DataFrame(
            data=top_spreader_potentials,
            index=top_spreaders,
            columns=network.y_names,
        ).sort_index()
        return df

    @staticmethod
    def convert_seed_set(
        seeds: list,
        actors_map: bidict,
    ) -> list[str]:
        """Convert indices of the actors_map into real names of actors from the network."""
        return [actors_map.inverse[seed] for seed in seeds]


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

        config["base"]['run_id'] = run_id
        evaluator = HeteroGNN_Predictor(config)
        result_path = Path(config["evaluation_dir"]) / 'results'
        result_path.mkdir(exist_ok=True, parents=True)

        evaluation_results = {}
        for idx, network in tqdm(enumerate(config["run"]["networks"])):
            tqdm.write(f'Processing: {network["name"]}')
            network_ts = evaluator(
                network_name=network["name"],
                network_type=network["type"],
            )

            network_ts.to_csv(
                result_path / f'{network["type"]}_{network["name"]}.csv'
            )
            evaluation_results[network["name"] + str(idx)] = {
                "features_type": config["data"]["features_type"],
                "network_ts": network_ts,
            }

        logging.info(evaluation_results)


if __name__ == "__main__":
    main()
