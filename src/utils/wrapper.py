from typing import Any, Iterator

from torch import cuda, optim
from torch.nn import MSELoss
from torch.nn.modules.loss import _Loss
from torch.nn.parameter import Parameter
from torch.optim import Optimizer, lr_scheduler
from torchmetrics.regression import MeanAbsoluteError

from src.training.loss.mtwae import MTWAE
from src.training.loss.slistmle import SListMLELoss
from src.training.loss.weighted_mse import WeightedMSE


def get_loss(
    loss_name: str,
    loss_args: dict[str, Any],
) -> _Loss:
    match loss_name:
        case MSELoss.__name__:
            return MSELoss(**loss_args)

        case WeightedMSE.__name__:
            return WeightedMSE(**loss_args)

        case MTWAE.__name__:
            return MTWAE(**loss_args)

        case MeanAbsoluteError.__name__:
            return MeanAbsoluteError(**loss_args)

        case SListMLELoss.__name__:
            return SListMLELoss(**loss_args)

        case _:
            raise AttributeError(f"Unknown loss function: {loss_name}")


def get_optimizer(
    optimizer_name: str,
    optimizer_args: dict[str, Any],
    model_parameters: Iterator[Parameter],
) -> Optimizer:
    try:
        cls = getattr(optim, optimizer_name)
    except AttributeError as e:
        raise AttributeError(f"Unknown optimizer: {optimizer_name}") from e

    return cls(params=model_parameters, **optimizer_args)


def get_scheduler(
    scheduler_name: str,
    scheduler_config: dict[str, Any],
    scheduler_args: dict[str, Any],
    optimizer: Optimizer,
) -> dict[str, Any]:
    try:
        cls = getattr(lr_scheduler, scheduler_name)
    except AttributeError as e:
        raise AttributeError(f"Unknown scheduler: {scheduler_name}") from e

    scheduler_config["scheduler"] = cls(
        optimizer=optimizer,
        **scheduler_args,
    )
    return scheduler_config


def get_accelerator(accelerator: str | None = None) -> str:
    if not accelerator:
        return "gpu" if cuda.is_available() else "cpu"
    return accelerator
