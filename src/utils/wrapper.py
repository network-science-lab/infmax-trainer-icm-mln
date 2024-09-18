from torch.nn import CrossEntropyLoss, MSELoss
from torch.nn.modules.loss import _Loss


def get_loss(loss_name: str) -> _Loss:
    match loss_name:
        case CrossEntropyLoss.__name__:
            return CrossEntropyLoss()

        case MSELoss.__name__:
            return MSELoss()

        case _:
            raise AttributeError(f"Unknown loss function: {loss_name}")
