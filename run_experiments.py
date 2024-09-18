"""Main entrypoint to the experiments."""

# TODO: consider adding runners and defaulf configs for each method
# TODO: change print statements to logs
import logging

import hydra
from omegaconf import DictConfig

from src import CONFIGS_PATH
from src.training.trainers.train import train
from src.training.trainers.utils import set_seed
from src.utils.config import get_available_configs, load_config


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
