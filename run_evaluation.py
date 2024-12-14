# TODO: refactor this sctipr due to changes in loaders!

import logging
import os
from pathlib import Path
from typing import Any

import hydra
import neptune
import network_diffusion as nd
import numpy as np
import torch
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
from src.wrapper.hetero import HetergoGNNWrapperConfig, HeteroGNNWrapper

load_dotenv(
    dotenv_path=Path(__file__).parent / ".env",
    override=True,
)


def weighted_sum(
    score: torch.Tensor,
    weights: torch.Tensor = torch.Tensor([4, 1, 1, 1]),
) -> torch.Tensor:
    return torch.sum(score * weights)


class HeteroGNN_Evaluator:
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
        network: nd.MultilayerNetworkTorch | None = None,
        **kwargs,
    ) -> np.ndarray:
        match self._config["base"]["selection_function"]:
            case "elimination_approach":
                return self.elimination_approach(network_type=network_type)
            case "top_k_approach":
                return self.top_k_approach(network_type=network_type)
            case _:
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
        )

        config["model_config"] = model_config["model"]

        return wrapper

    def elimination_approach(self, network_type: str) -> np.ndarray:
        ow = torch.Tensor(
            [
                self._config["data"]["output_weights"]["w_e"],
                self._config["data"]["output_weights"]["w_sl"],
                self._config["data"]["output_weights"]["w_pit"],
                self._config["data"]["output_weights"]["w_pin"],
            ]
        )

        net = load_network(net_name=network_type, as_tensor=False)[network_type]
        sp_df = load_sp(net_name=network_type)[network_type]

        top_spreader = None
        result = []
        for _ in range(self._config["base"]["nb_seeds"]):
            not_ts_actors = [
                actor for actor in net.get_actors() if actor.actor_id != top_spreader
            ]
            net = net.subgraph(not_ts_actors)
            sp_df = sp_df[sp_df["actor"] != str(top_spreader)]

            mln_info = MLNInfo(
                mln_type=network_type,
                mln_name=network_type,
                mln=net,
                icm_protocol=self._config["data"]["protocol"],
                x_type=self._config["data"]["features_type"],
                y_type=self._config["data"]["output_label_name"],
                sp_raw=sp_df,
                icm_p=self._config["base"]["icm_p"],
            )
            network = MLNHeteroData.from_network_info(
                network_info=mln_info,
                output_dim=self._config["model_config"]["parameters"]["output_dim"],
                input_dim=self._config["model_config"]["parameters"]["input_dim"],
            )

            data = self._wrapper.predict_step(
                batch=network,
                batch_idx=0,
            )

            weighted_sums = {
                k: weighted_sum(
                    score=v,
                    weights=ow,
                )
                for k, v in data["actor"].items()
            }
            max_key = max(weighted_sums, key=weighted_sums.get)

            top_spreader = next(
                (k for k, v in mln_info.mln_torch.actors_map.items() if v == max_key),
                None,
            )
            result.append(top_spreader)

        return np.asarray(result)

    def top_k_approach(self, network_type: str) -> np.ndarray:
        ow = torch.Tensor(
            [
                self._config["data"]["output_weights"]["w_e"],
                self._config["data"]["output_weights"]["w_sl"],
                self._config["data"]["output_weights"]["w_pit"],
                self._config["data"]["output_weights"]["w_pin"],
            ]
        )

        net = load_network(net_name=network_type, as_tensor=False)[network_type]
        sp_df = load_sp(net_name=network_type)[network_type]

        mln_info = MLNInfo(
            mln_type=network_type,
            mln_name=network_type,
            mln=net,
            icm_protocol=self._config["data"]["protocol"],
            x_type=self._config["data"]["features_type"],
            y_type=self._config["data"]["output_label_name"],
            sp_raw=sp_df,
            icm_p=self._config["base"]["icm_p"],
        )
        network = MLNHeteroData.from_network_info(
            network_info=mln_info,
            output_dim=self._config["model_config"]["parameters"]["output_dim"],
            input_dim=self._config["model_config"]["parameters"]["input_dim"],
        )

        data = self._wrapper.predict_step(
            batch=network,
            batch_idx=0,
        )

        weighted_sums = {
            k: weighted_sum(
                score=v,
                weights=ow,
            )
            for k, v in data["actor"].items()
        }
        sorted_actors = sorted(
            weighted_sums,
            key=weighted_sums.get,
            reverse=True,
        )
        sorted_actors = sorted_actors[: self._config["base"]["nb_seeds"]]
        top_spreaders = [
            k for k, v in mln_info.mln_torch.actors_map.items() if v in sorted_actors
        ]

        return np.asarray(top_spreaders)


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

    evaluator = HeteroGNN_Evaluator(config)

    evaluation_results = {}
    for idx, network in tqdm(enumerate(config["run"]["networks"])):
        network_ts = evaluator(
            network_type=network["name"],
        )

        evaluation_results[network["name"] + str(idx)] = {
            "features_type": config["data"]["features_type"],
            "network_ts": network_ts,
        }
        idx += 1

    logging.info(evaluation_results)


if __name__ == "__main__":
    main()
