from typing import Any, Callable

from src.infmax_models.hetero_gat_conv import GATHeteroGNN
from src.infmax_models.ssnet.ssnet import SSNet


def load_model(config: dict[str, Any]) -> Callable:
    """Load and initialize infmax model."""
    model_name = config["model"]["name"]
    model_params = config["model"]["parameters"]

    match model_name:

        case GATHeteroGNN.__name__:
            return GATHeteroGNN(**model_params)

        case SSNet.__name__:
            return SSNet(**model_params)

        case _:
            raise AttributeError(f"Unknown model: {model_name}")
