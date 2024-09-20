from typing import Any, Callable

from src.infmax_models.hetero_gat_conv import GATHeteroGNN
from src.infmax_models.multi_node2vec_kmeans import (MultiNode2VecKMeans,
                                                     MultiNode2VecKMeansAuto)


def load_model(config: dict[str, Any]) -> Callable:
    """Load and initialize infmax model."""
    model_name = config["model"]["name"]
    model_params = config["model"]["parameters"]

    match model_name:
        case MultiNode2VecKMeans.__name__:
            model_params["k_means"]["nb_seeds"] = config["train"]["seed_size"]
            model_params["k_means"]["random_state"] = config["base"]["random_seed"]
            model_params["multi_node2vec"]["random_state"] = config["base"][
                "random_seed"
            ]
            return MultiNode2VecKMeans(**model_params)

        case MultiNode2VecKMeansAuto.__name__:
            model_params["k_means"]["nb_seeds"] = config["train"]["seed_size"]
            model_params["k_means"]["random_state"] = config["base"]["random_seed"]
            model_params["multi_node2vec"]["random_state"] = config["base"][
                "random_seed"
            ]
            return MultiNode2VecKMeansAuto(**model_params)

        case GATHeteroGNN.__name__:
            return GATHeteroGNN(**model_params)

        case _:
            raise AttributeError(f"Unknown model: {model_name}")
