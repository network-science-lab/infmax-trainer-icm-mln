
from typing import Callable
from ss_models.multi_node2vec_kmeans import MultiNode2VecKMeans


def load_model(model_config: str) -> Callable:
    model_name = model_config["name"]
    model_params = model_config["parameters"]
    if model_name == "MultiNode2VecKMeans":
        return MultiNode2VecKMeans(**model_params)
    # TODO: here go other models
    raise AttributeError(f"Unknown model: {model_name}")
