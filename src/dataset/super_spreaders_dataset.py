from torch_geometric.data import Dataset
from torch_geometric.transforms import BaseTransform
from torch_geometric.typing import EdgeType, NodeType

from src.data_models.mln_hetero_data import MLNHeteroData
from src.data_models.mln_info import MLNInfo


class SuperSpreadersDataset(Dataset):
    """Class handling the main dataset."""

    def __init__(
        self,
        networks: list[MLNInfo],
        input_dim: int,
        output_dim: int,
        transform: BaseTransform | None,
    ) -> None:
        """
        Initialise the object.

        :param networks: list of objects containing information for creating hetero data.
        :param input_dim: number of features for h0 vector
        :param output_dim: number of output classes created during discretization (if any).
        :param transform: method to transform features
        """
        super().__init__()
        self._input_dim = input_dim
        self._output_dim = output_dim
        self.data_list = networks
        self.transform = transform

    def len(self):
        return len(self.data_list)

    def get(self, idx) -> MLNHeteroData:
        mln_hetero_data = MLNHeteroData.from_network_info(
            network_info=self.data_list[idx],
            input_dim=self._input_dim,
            output_dim=self._output_dim,
        )
        if self.transform:
            return self.transform(mln_hetero_data)
        return mln_hetero_data
