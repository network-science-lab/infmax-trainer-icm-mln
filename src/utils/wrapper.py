from typing import Any, Iterator

from torch import cuda
from torch.nn import CrossEntropyLoss, MSELoss
from torch.nn.modules.loss import _Loss
from torch.optim import Optimizer
from torch import optim
from torch.optim import lr_scheduler
from torch.nn.parameter import Parameter

from src.training.loss.weighted_mse import WeightedMSE
from src.training.loss.mrtwse import MRTWSE


def get_loss(
    loss_name: str,
    loss_args: dict[str, Any],
) -> _Loss:
    match loss_name:
        case CrossEntropyLoss.__name__:
            return CrossEntropyLoss(**loss_args)

        case MSELoss.__name__:
            return MSELoss(**loss_args)

        case WeightedMSE.__name__:
            return WeightedMSE(**loss_args)

        case MRTWSE.__name__:
            return MRTWSE(**loss_args)

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

def get_accelerator(accelerator: str | None) -> str:
    if not accelerator:
        return "gpu" if cuda.is_available() else "cpu"
    return accelerator
