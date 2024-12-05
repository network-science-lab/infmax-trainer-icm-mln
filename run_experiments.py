"""Main entrypoint to the experiments."""

# TODO: change p_value in the codebase so that its name is related with ICM not statistics

import logging
from pathlib import Path

import hydra
from dotenv import load_dotenv
from omegaconf import DictConfig

from src import CONFIGS_PATH
from src.training.trainer import train
from src.utils.config import get_available_configs, load_config
from src.utils.seed import set_seed

load_dotenv(
    dotenv_path=Path(__file__).parent / ".env",
    override=True,
)


@hydra.main(
    version_base=None,
    config_path=str(CONFIGS_PATH),
    config_name="hydra",
)
def main(cfg: DictConfig) -> None:
    if cfg["experiment-config"] not in get_available_configs():
        raise ValueError(
            f"Invalid option: {cfg['experiment-config']}. Choose from {get_available_configs()}"
        )

    config = load_config(
        cfg=cfg,
        cofig_path=CONFIGS_PATH / f"{cfg['experiment-config']}.yaml",
    )
    if random_seed := config["base"].get("random_seed"):
        logging.info(f"Setting randomness seed as {random_seed}!")
        set_seed(config["base"]["random_seed"])

    logging.info(f"Loaded config: {config}")
    train(config)


if __name__ == "__main__":
    main()
