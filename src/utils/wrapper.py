from typing import Any, Iterator

from torch import cuda
from torch.nn import CrossEntropyLoss, MSELoss
from torch.nn.modules.loss import _Loss
from torch.optim import Optimizer, Adam, AdamW
from torch.optim.lr_scheduler import ReduceLROnPlateau
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
    match optimizer_name:
        case Adam.__name__:
            return Adam(params=model_parameters, **optimizer_args)

        case AdamW.__name__:
            return AdamW(params=model_parameters, **optimizer_args)

        case _:
            raise AttributeError(f"Unknown optimizer: {optimizer_name}")
        
def get_scheduler(
    scheduler_name: str,
    scheduler_config: dict[str, Any],
    scheduler_args: dict[str, Any],
    optimizer: Optimizer,
) -> dict[str, Any]:
    match scheduler_name:
        case ReduceLROnPlateau.__name__:
            scheduler_config["scheduler"] = ReduceLROnPlateau(
                optimizer=optimizer, 
                **scheduler_args,
            )
            return scheduler_config

        case _:
            raise AttributeError(f"Unknown scheduler: {scheduler_name}")
        

def get_device(device: str | list[int]) -> str:
    if device == 'auto':
        return 'cuda:0' if cuda.is_available() else 'cpu' 
    elif type(device) == list:
        return f'cuda:{device[0]}'
        