import ast
import logging
import os
import warnings
from pathlib import Path
from typing import Any, Callable

import hydra
import neptune
import pandas as pd
from dotenv import load_dotenv
from omegaconf import DictConfig
from tqdm import tqdm

from _data_set.nsl_data_utils.loaders.constants import (
    EXPOSED,
    PEAK_INFECTED,
    PEAK_ITERATION,
    SIMULATION_LENGTH,
)
from src import CONFIGS_PATH
from src.data_models.mln_info import MLNInfo
from src.datamodule.loader import get_transform
from src.dataset.super_spreaders_dataset import SuperSpreadersDataset
from src.infmax_models.loader import load_model
from src.utils.config import load_config
from src.utils.misc import set_seed
from src.wrapper.mln_hetero import HetergoGNNWrapperConfig, HeteroGNNWrapper

load_dotenv(
    dotenv_path=Path(__file__).parent / ".env",
    override=True,
)


class HeteroGNN_Predictor:

    def __init__(self, config: dict[str, Any]) -> None:
        self._eval_config = config

        evaluation_dir = Path(f'_data_results/evaluation/{config["base"]["run_id"]}')
        evaluation_dir.mkdir(exist_ok=True, parents=True)
        self.evaluation_dir = evaluation_dir

        if random_seed := config["base"].get("random_seed"):
            logging.info(f"Setting randomness seed as {random_seed}!")
            set_seed(config["base"]["random_seed"])

        self.run = neptune.init_run(
            api_token=os.getenv(
                key="NEPTUNE_API_KEY",
                default=neptune.ANONYMOUS_API_TOKEN,
            ),
            project=config["base"]["project"],
            with_id=config["base"]["run_id"],
        )
        wrapper_obj, wrapper_config = self.from_neptune(config)
        self._wrapper_obj = wrapper_obj
        self._wrapper_obj.eval()
        self._wrapper_config = wrapper_config

    def __call__(self, *args, **kwargs) -> pd.DataFrame:
        return self.top_k_approach(*args, **kwargs)

    def upload_result(
        self, network_type: str, network_name: str, result_path: Path
    ) -> None:
        self.run[f"evaluation/{network_type}/{network_name}"].upload(str(result_path))

    def from_neptune(
        self, run_config: dict[str, Any]
    ) -> tuple[HeteroGNNWrapper, dict[str, Any]]:
        temp = Path(self.run["training/model/best_model_path"].fetch())
        neptune_best_ckpt_path = f"training/model/checkpoints/{temp.stem}"
        local_best_ckpt_path = str(self.evaluation_dir / "best.ckpt")
        self.run[neptune_best_ckpt_path].download(destination=local_best_ckpt_path)

        model_config = {
            "model": {
                "name": self.run["training/hyperparams/model/name"].fetch(),
                "parameters": self.run["training/hyperparams/model/parameters"].fetch(),
            }
        }
        model = load_model(model_config)

        wrapper_config = HetergoGNNWrapperConfig.from_str(
            self.run["training/hyperparams/config"].fetch()
        )
        wrapper = HeteroGNNWrapper.load_from_checkpoint(
            checkpoint_path=local_best_ckpt_path,
            model=model,
            config=wrapper_config,
        ).to(run_config["base"]["device"])

        run_config = {
            "model": model_config["model"],
            "data": self.run["training/hyperparams/data"].fetch(),
        }

        return wrapper, run_config

    def prepare_dataset(
        self, network_type: str, network_name: str
    ) -> SuperSpreadersDataset:
        mln_info = MLNInfo.from_config(
            mln_type=network_type,
            mln_name=network_name,
            icm_protocol=self._wrapper_config["data"]["icm"]["protocol"],
            icm_p=self._wrapper_config["data"]["icm"]["p"],
            x_type=ast.literal_eval(self._wrapper_config["data"]["train_data"])[0][
                "features_type"
            ],
            y_type=ast.literal_eval(self._wrapper_config["data"]["output_label_name"]),
        )
        transform = get_transform(self._wrapper_config["data"]["transform"])
        return SuperSpreadersDataset(
            networks=[mln_info],
            input_dim=self._wrapper_config["model"]["parameters"]["input_dim"],
            output_dim=self._wrapper_config["model"]["parameters"]["output_dim"],
            transform=transform,
        )

    def top_k_approach(self, network_type: str, network_name: str) -> pd.DataFrame:
        dataset = self.prepare_dataset(network_type, network_name)
        mln_hetero_data = dataset[0]
        mln_hetero_data.to(self._eval_config["base"]["device"])
        with warnings.catch_warnings(
            record=True
        ) as _:  # it's due to an attempt to log to neptune
            self._wrapper_obj.test_step(batch=mln_hetero_data, batch_idx=0)
        prediction_raw = self._wrapper_obj.test_preds["preds"][
            f"{mln_hetero_data.network_type[0]}_{mln_hetero_data.network_name[0]}"
        ]
        prediction_sorted = prediction_raw.sort_values(
            [EXPOSED, SIMULATION_LENGTH, PEAK_INFECTED, PEAK_ITERATION],
            ascending=[False, True, True, False],
        )
        return prediction_sorted


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

    for run_id in config["base"]["run_ids"]:
        logging.info(f"Run: {run_id}")

        config["base"]["run_id"] = run_id["id"]

        evaluator = HeteroGNN_Predictor(config)
        result_path = evaluator.evaluation_dir / "results"
        result_path.mkdir(exist_ok=True, parents=True)

        for network in tqdm(config["run"]["networks"]):
            tqdm.write(f'Processing: {network["name"]}')
            network_ts = evaluator(
                network_name=network["name"], network_type=network["type"]
            )
            network_ts.to_csv(result_path / f"{network['type']}_{network['name']}.csv")
            evaluator.upload_result(
                network_name=network["name"],
                network_type=network["type"],
                result_path=result_path / f"{network['type']}_{network['name']}.csv",
            )
        logging.info("Evaluation completed")


if __name__ == "__main__":
    main()
