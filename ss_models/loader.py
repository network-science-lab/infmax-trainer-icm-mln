
from typing import Callable
from ss_models.multi_node2vec_kmeans import MultiNode2VecKMeans


def load_model(model_name: str) -> Callable:
    if model_name == "MultiNode2VecKMeans":
        return MultiNode2VecKMeans
    # TODO: here goes another models
    raise AttributeError(f"Unknown model: {model_name}")
