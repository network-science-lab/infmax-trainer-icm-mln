from abc import abstractmethod

from torch import Tensor
from torch.fx import Proxy
from torch.nn import Module


class BaseHeteroModule(Module):
    def __init__(
        self,
    ) -> None:
        super().__init__()

    @abstractmethod
    def forward(
        self,
        x_dict: Proxy,
        z_dict: Proxy,
        edge_index_dict: Proxy,
    ) -> dict[str, Tensor]:
        pass
