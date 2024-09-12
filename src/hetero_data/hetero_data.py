from typing import Iterable

from torch_geometric.data import HeteroData


class LightningHeteroData(HeteroData):
    def __iter__(self) -> Iterable:
        for node_type, data in self.node_items():
            for key, value in data.items():
                yield node_type, key, value

        for edge_type, data in self.edge_items():
            for key, value in data.items():
                yield edge_type, key, value
