from typing import Callable

from torch_geometric.data import Dataset, HeteroData
from torch_geometric.typing import EdgeType, NodeType

from src.hetero_data.hetero_data import LightningHeteroData
from src.utils.multilayer_network import MultilayerNetworkInfo


class BaseHeteroDataset(Dataset):
    r"""Base Dataset class for creating graph datasets based on hetero data.

    Args:
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

        self.data_list: list[LightningHeteroData]

    def len(self) -> int:
        return len(self.data_list)

    def get(self, idx) -> HeteroData:
        return self.data_list[idx]

    def get_metadata(self) -> tuple[list[NodeType], list[EdgeType]]:
        nodes_data = []
        edges_data = []
        for data in self.data_list:
            node_data, edge_data = data.metadata()
            nodes_data.extend(node_data)
            edges_data.extend(edge_data)

        return nodes_data, edges_data
