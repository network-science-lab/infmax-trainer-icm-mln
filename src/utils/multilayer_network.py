from dataclasses import dataclass

from network_diffusion.mln import MultilayerNetworkTorch
from pandas import DataFrame


@dataclass(frozen=True)
class MultilayerNetworkInfo:
    network: MultilayerNetworkTorch
    network_name: str
    protocol: str | None
    features_type: str | None
    output_label_name: str | list[str] | None
    spreading_potential: DataFrame | None
