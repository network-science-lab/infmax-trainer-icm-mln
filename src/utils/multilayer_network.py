from dataclasses import dataclass

from network_diffusion.mln import MultilayerNetworkTorch
from network_diffusion.mln.mlnetwork import MultilayerNetwork
from pandas import DataFrame


@dataclass
class MultilayerNetworkInfo:
    label_name: str
    network: MultilayerNetwork | MultilayerNetworkTorch
    output_df: DataFrame
    network_name: str

    def __post_init__(self):
        if type(self.network) is MultilayerNetwork:
            self.network = MultilayerNetworkTorch.from_mln(self.network)
