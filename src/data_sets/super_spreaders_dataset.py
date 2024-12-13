from torch_geometric.data import HeteroData

from src.data_sets.base_dataset import BaseDataSet
from src.data_models.mln_hetero_data import MLNHeteroData
from src.data_models.mln_info import MLNInfo


class SuperSpreadersDataSet(BaseDataSet):
    """Class handling the main dataset."""

    def __init__(
        self,
        networks: list[MLNInfo],
        # input_dim: int,
        # output_dim: int,
        root: str,
    ) -> None:
        """
        Initialise the object.

        :param networks: list of objects containing information for creating hetero data.
        :param input_dim: number of features for h0 vector
        :param output_dim: number of output classes created during discretization (if any).
        :param root: required by parent, a root path of the dataset
        """
        super().__init__(root=root)
        # self._input_dim = input_dim
        # self._output_dim = output_dim
        self.data_list = networks

        # self.data_list = [
        #     MLNHeteroData.from_network_info(
        #         network_info=network_info,
        #         output_dim=output_dim,
        #         input_dim=input_dim,
        #     )
        #     for network_info in networks
        # ]

    def len(self) -> int:
        return len(self.data_list)

    def get(self, idx) -> HeteroData:
        return self.data_list[idx]
