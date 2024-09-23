from dataclasses import dataclass

from network_diffusion.mln import MultilayerNetworkTorch
from network_diffusion.mln.mlnetwork import MultilayerNetwork
from pandas import DataFrame


# TODO: ADD different feature generation
@dataclass
class MultilayerNetworkInfo:
    network: MultilayerNetwork | MultilayerNetworkTorch
    network_name: str
    protocol: str | None
    output_label_name: str | list[str] | None
    spreading_potential: DataFrame | None

    def __post_init__(self):
        if type(self.network) is MultilayerNetwork:
            self.network = MultilayerNetworkTorch.from_mln(self.network)
