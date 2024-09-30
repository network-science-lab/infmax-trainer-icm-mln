from typing import Callable

from torch_geometric.data import HeteroData

from src.dataset.base_hetero_dataset import BaseHeteroDataset
from src.hetero_data.hetero_data import LightningHeteroData
from src.utils.multilayer_network import MultilayerNetworkInfo


class DataFrameHeteroDataset(BaseHeteroDataset):
    r"""Dataset class for creating graph datasets based on hetero data.

    Args:
        networks (list[MultilayerNetworkInfo]): List of objects containing
            information for creating hetero data.
        input_dim (int): Size of features for h0 vector.
        output_dim (int): Number of output classes created during
            discretization.
        root (str, optional): Root directory where the dataset should be saved.
            (optional: :obj:`None`)
        transform (callable, optional): A function/transform that takes in a
            :class:`~torch_geometric.data.Data` or
            :class:`~torch_geometric.data.HeteroData` object and returns a
            transformed version.
            The data object will be transformed before every access.
            (default: :obj:`None`)
        pre_transform (callable, optional): A function/transform that takes in
            a :class:`~torch_geometric.data.Data` or
            :class:`~torch_geometric.data.HeteroData` object and returns a
            transformed version.
            The data object will be transformed before being saved to disk.
            (default: :obj:`None`)
        pre_filter (callable, optional): A function that takes in a
            :class:`~torch_geometric.data.Data` or
            :class:`~torch_geometric.data.HeteroData` object and returns a
            boolean value, indicating whether the data object should be
            included in the final dataset. (default: :obj:`None`)
    """

    def __init__(
        self,
        networks: list[MultilayerNetworkInfo],
        input_dim: int,
        output_dim: int,
        root: str,
        transform: Callable | None = None,
        pre_filter: Callable | None = None,
        pre_transform: Callable | None = None,
    ) -> None:
        super().__init__(
            root=root,
            transform=transform,
            pre_transform=pre_transform,
            pre_filter=pre_filter,
        )

        self._input_dim = input_dim
        self._output_dim = output_dim

        self.data_list = [
            LightningHeteroData.from_network_info(
                network_info=network_info,
                output_dim=output_dim,
                input_dim=input_dim,
            )
            for network_info in networks
        ]

    def len(self) -> int:
        return len(self.data_list)

    def get(self, idx) -> HeteroData:
        return self.data_list[idx]
