from dataclasses import dataclass

from network_diffusion.mln import MultilayerNetworkTorch
from network_diffusion.mln.mlnetwork import MultilayerNetwork
from pandas import DataFrame


@dataclass
class MultilayerNetworkInfo:
    network: MultilayerNetwork | MultilayerNetworkTorch
    network_name: str
    protocol: str | None
    features_type: str | None
    output_label_name: str | list[str] | None
    spreading_potential: DataFrame | None
    mln_network: MultilayerNetwork | None = None

    def __post_init__(self):
        if type(self.network) is MultilayerNetwork:
            self.mln_network = self.network
            self.network = MultilayerNetworkTorch.from_mln(self.network)
