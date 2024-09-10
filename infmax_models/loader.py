
from typing import Any, Callable
from infmax_models.multi_node2vec_kmeans import MultiNode2VecKMeans, MultiNode2VecKMeansAuto


def load_model(model_config: dict[str, Any], train_config: dict[str, Any]) -> Callable:
    """Load and initialize infmax model."""
    model_name = model_config["name"]
    model_params = model_config["parameters"]

    match model_name:
        case "MultiNode2VecKMeans":
            model_params["k_means"]["nb_seeds"] = train_config["seed_size"]
            model_params["k_means"]["random_state"] = train_config["random_seed"]
            model_params["multi_node2vec"]["random_state"] = train_config["random_seed"]
            return MultiNode2VecKMeans(**model_params)

        case "MultiNode2VecKMeansAuto":
            model_params["k_means"]["nb_seeds"] = train_config["seed_size"]
            model_params["k_means"]["random_state"] = train_config["random_seed"]
            model_params["multi_node2vec"]["random_state"] = train_config["random_seed"]
            return MultiNode2VecKMeansAuto(**model_params)

        case _:
            raise AttributeError(f"Unknown model: {model_name}")
