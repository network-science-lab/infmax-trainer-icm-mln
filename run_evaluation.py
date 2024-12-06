import logging
import os
from pathlib import Path
from typing import Any

import hydra
import neptune
import numpy as np
import torch
from dotenv import load_dotenv
from omegaconf import DictConfig

from _data_set.nsl_data_utils.loaders.net_loader import load_network
from _data_set.nsl_data_utils.loaders.sp_loader import load_sp
from src import CONFIGS_PATH
from src.data_models.mln_hetero_data import MLNHeteroData
from src.data_models.mln_info import MLNInfo
from src.infmax_models.loader import load_model
from src.utils.config import load_config
from src.utils.seed import set_seed
from src.wrapper.hetero import HetergoGNN_WrapperConfig, HeteroGNN_Wrapper

load_dotenv(
    dotenv_path=Path(__file__).parent / ".env",
    override=True,
)


def _load_model_from_neptune(config: dict[str, Any]) -> HeteroGNN_Wrapper:
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
    local_best_ckpt_path = f"{config['hydra']['run']['dir']}/best.ckpt"
    run[neptune_best_ckpt_path].download(destination=local_best_ckpt_path)

    model_config = {
        "model": {
            "name": run["training/hyperparams/model/name"].fetch(),
            "parameters": run["training/hyperparams/model/parameters"].fetch(),
        }
    }
    model = load_model(model_config)
    wrapper_config: HetergoGNN_WrapperConfig = eval(
        run["training/hyperparams/config"].fetch()
    )
    wrapper = HeteroGNN_Wrapper.load_from_checkpoint(
        checkpoint_path=local_best_ckpt_path,
        model=model,
        config=wrapper_config,
    )

    config["model_config"] = model_config["model"]

    return wrapper


def weighted_sum(
    score: torch.Tensor,
    weights: torch.Tensor = torch.Tensor([4, 1, 1, 1]),
) -> torch.Tensor:
    return torch.sum(score * weights)


def get_top_spreaders(
    network_type: str,
    features_type: str,
    config: dict[str, Any],
    wrapper: HeteroGNN_Wrapper,
) -> np.ndarray:
    model_config = config["model_config"]

    net = load_network(net_name=network_type, as_tensor=False)[network_type]
    sp_df = load_sp(net_name=network_type)[network_type]

    top_spreader = None
    result = []

    for _ in range(config["base"]["nb_seeds"]):
        not_ts_actors = [
            actor for actor in net.get_actors() if actor.actor_id != top_spreader
        ]
        net = net.subgraph(not_ts_actors)
        sp_df = sp_df[sp_df["actor"] != str(top_spreader)]

        mln_info = MLNInfo(
            mln_type=network_type,
            mln_name=network_type,
            mln=net,
            icm_protocol=config["data"]["protocol"],
            x_type=features_type,
            y_type=config["data"]["output_label_name"],
            sp_raw=sp_df,
        )
        network = MLNHeteroData.from_network_info(
            network_info=mln_info,
            output_dim=model_config["parameters"]["output_dim"],
            input_dim=model_config["parameters"]["input_dim"],
        )

        data = wrapper.predict_step(
            batch=network,
            batch_idx=0,
        )

        weighted_sums = {k: weighted_sum(v) for k, v in data["actor"].items()}
        max_key = max(weighted_sums, key=weighted_sums.get)

        top_spreader = next(
            (k for k, v in mln_info.mln_torch.actors_map.items() if v == max_key), None
        )
        result.append(top_spreader)

    return np.asarray(result)


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
    if random_seed := config["base"].get("random_seed"):
        logging.info(f"Setting randomness seed as {random_seed}!")
        set_seed(config["base"]["random_seed"])

    logging.info(f"Loaded config: {config}")

    wrapper = _load_model_from_neptune(config)
    wrapper.eval()

    evaluation_results = {}
    idx = 0
    for network in config["data"]["networks"]:
        network_ts = get_top_spreaders(
            network_type=network["name"],
            features_type=network["features_type"],
            config=config,
            wrapper=wrapper,
        )

        evaluation_results[network["name"] + str(idx)] = {
            "features_type": network["features_type"],
            "network_ts": network_ts,
        }
        idx += 1

    # TODO: CHECK WHETER MODEL IS DETERMINISTIC
    pass

if __name__ == "__main__":
    main()
