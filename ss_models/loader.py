
from typing import Callable

from ss_models.multinode2vec_kmeans import Multi_Node2Vec_KMeans


def load_model(model_name: str) -> Callable:
    if model_name == "multinode2vec_kmeans":
        return Multi_Node2Vec_KMeans
    # TODO: here goes another models
    raise AttributeError(f"Unknown model: {model_name}")
