from typing import Any

from torch.nn import CrossEntropyLoss, MSELoss
from torch.nn.modules.loss import _Loss


def get_loss(
    loss_name: str,
    loss_args: dict[str, Any],
) -> _Loss:
    match loss_name:
        case CrossEntropyLoss.__name__:
            return CrossEntropyLoss(**loss_args)

        case MSELoss.__name__:
            return MSELoss(**loss_args)

        case _:
            raise AttributeError(f"Unknown loss function: {loss_name}")
