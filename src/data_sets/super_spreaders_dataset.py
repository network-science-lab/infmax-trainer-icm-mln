from torch_geometric.data import HeteroData

from src.data_sets.base_dataset import BaseDataSet
from src.data_models.mln_hetero_data import MLNHeteroData
from src.data_models.mln_info import MLNInfo

import torch
from torch.utils.data import Dataset

class SuperSpreadersDataSet(Dataset):
    """Class handling the main dataset."""

    def __init__(self, networks: list[MLNInfo], input_dim: int, output_dim: int) -> None:
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

        # self.data_list = [
        #     MLNHeteroData.from_network_info(
        #         network_info=network_info,
        #         output_dim=output_dim,
        #         input_dim=input_dim,
        #     )
        #     for network_info in networks
        # ]
    
    def __len__(self):
        return len(self.data_list)

    def __getitem__(self, idx) -> HeteroData:
        return MLNHeteroData.from_network_info(self.data_list[idx], self._input_dim, self._output_dim)
