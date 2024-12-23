from torch_geometric.data import Dataset
from torch_geometric.typing import EdgeType, NodeType

from src.data_models.mln_hetero_data import MLNHeteroData
from src.data_models.mln_info import MLNInfo


class SuperSpreadersDataset(Dataset):
    """Class handling the main dataset."""

    def __init__(
        self, networks: list[MLNInfo], input_dim: int, output_dim: int
    ) -> None:
        """
        Initialise the object.

        :param networks: list of objects containing information for creating hetero data.
        :param input_dim: number of features for h0 vector
        :param output_dim: number of output classes created during discretization (if any).
        """
        super().__init__()
        self._input_dim = input_dim
        self._output_dim = output_dim
        self.data_list = networks

        # self._metadata = self._prepare_metadata() # TODO: UNCOMMENT TS NETWORK
        self._metadata = ([], [])

    def len(self):
        return len(self.data_list)

    def get(self, idx) -> MLNHeteroData:
        return MLNHeteroData.from_network_info(
            network_info=self.data_list[idx],
            input_dim=self._input_dim,
            output_dim=self._output_dim,
        )

    def _prepare_metadata(self) -> tuple[list[NodeType], list[EdgeType]]:
        nodes_data = []
        edges_data = []
        for i in range(self.data_list):
            node_data, edge_data = self.get(i).metadata()
            nodes_data.extend(node_data)
            edges_data.extend(edge_data)

        return nodes_data, edges_data

    def get_metadata(self) -> tuple[list[NodeType], list[EdgeType]]:
        return self._metadata
