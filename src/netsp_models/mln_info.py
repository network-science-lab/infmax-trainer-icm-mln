from dataclasses import dataclass
from typing import Literal

from network_diffusion.mln import MultilayerNetwork, MultilayerNetworkTorch, functions
from pandas import DataFrame

from _data_set.nsl_data_utils.loaders.constants import (
    ACTOR, AND, EXPOSED, OR, PEAK_INFECTED, PEAK_ITERATION, PROTOCOL, SIMULATION_LENGTH
)


@dataclass(frozen=True)
class MultilayerNetworkInfo:
    """Base class to keep network with its ground truth data and prediciton params."""

    name: str  # its type and name, e.g. aucs_aucs artificial_small_er2
    net_nd: MultilayerNetwork
    icm_protocol: Literal["OR", "AND"]
    x_type: Literal["zeros", "scrapped", "centralities"]
    y_type: list[Literal["exposed", "simulation_length", "peak_iteration", "peak_infected"]]
    sp_raw: DataFrame
    net_pt: MultilayerNetworkTorch | None = None

    def __post_init__(self) -> None:
        assert self.icm_protocol in {OR, AND}
        assert self.x_type in {"zeros", "scrapped", "centralities"}
        assert set(self.y_type).issubset({EXPOSED, SIMULATION_LENGTH, PEAK_ITERATION, PEAK_INFECTED})
        object.__setattr__(self, 'net_nd', functions.remove_selfloop_edges(self.net_nd))
        object.__setattr__(self, 'net_pt', MultilayerNetworkTorch.from_mln(self.net_nd))

    def transform_sp(self) -> DataFrame:
        """Transform DataFrame with raw spreading potentials according to given parameters."""
        spreading_potential_columns = [ACTOR, PROTOCOL, *self.y_type]
        df = self.sp_raw[spreading_potential_columns]
        df = df[df[PROTOCOL] == self.icm_protocol].drop(PROTOCOL, axis=1)
        df = df.groupby([ACTOR]).mean().reset_index()
        return df
