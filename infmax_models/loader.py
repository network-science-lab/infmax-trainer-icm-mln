
from typing import Any, Callable
from infmax_models.multi_node2vec_kmeans import MultiNode2VecKMeans


def load_model(model_config: dict[str, Any], train_config: dict[str, Any]) -> Callable:
    """Load and initialise infmax model."""
    model_name = model_config["name"]
    model_params = model_config["parameters"]

    if model_name == "MultiNode2VecKMeans":
        model_params["k_means"]["num_segments"] = train_config["seed_size"]
        model_params["k_means"]["random_state"] = train_config["random_seed"]
        return MultiNode2VecKMeans(**model_params)
    elif 1 == 1:
        # TODO: here go other models
        print("add mode models, please!")

    raise AttributeError(f"Unknown model: {model_name}")
