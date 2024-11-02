from abc import abstractmethod

from torch import Tensor
from torch.fx import Proxy
from torch.nn import Module


class BaseHeteroModule(Module):
    def __init__(
        self,
        is_hetero: bool,
    ) -> None:
        super().__init__()
        self.is_hetero = is_hetero

    @abstractmethod
    def forward(
        self,
        x_dict: Proxy,
        z_dict: Proxy,
        edge_index_dict: Proxy,
    ) -> dict[str, Tensor]:
        pass
