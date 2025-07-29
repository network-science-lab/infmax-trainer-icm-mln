import random

import numpy as np
import torch


def set_seed(seed):
    """Fix seeds for reproducable experiments."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def general_test_result(test_output: dict[str, float]) -> dict[str, float]:
    """Compute std and avg loss over all tested networks."""
    losses = np.array(
        [
            test_value
            for test_key, test_value in test_output[0].items()
            if "loss" in test_key
        ]
    )
    return {"AVG": losses.mean(), "STD": losses.std()}
