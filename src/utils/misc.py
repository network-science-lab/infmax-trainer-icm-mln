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
    losses = np.array([test_loss_value for test_loss_value in test_output[0].values()])
    # losses = np.fromiter(test_output[0].values(), dtype=np.float32)  # TODO: test this as well
    return {"AVG": losses.mean(), "STD": losses.std()}
