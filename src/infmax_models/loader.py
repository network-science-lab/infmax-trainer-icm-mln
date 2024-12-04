from typing import Any, Callable

from src import infmax_models


def load_model(config: dict[str, Any]) -> Callable:
    """Load and initialize infmax model."""
    model_name = config["model"]["name"]
    model_params = config["model"]["parameters"]

    try:
        cls = getattr(infmax_models, model_name)
    except AttributeError as e:
        raise ValueError(f"Unknown model: {model_name}") from e

    return cls(**model_params)
