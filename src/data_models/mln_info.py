"""A script where auxiliary data model is implemented."""

from dataclasses import dataclass
from typing import Literal

from network_diffusion.mln import MultilayerNetwork, MultilayerNetworkTorch, functions
from pandas import DataFrame

from _data_set.nsl_data_utils.loaders.constants import (
    ACTOR, AND, EXPOSED, OR, PEAK_INFECTED, PEAK_ITERATION, PROTOCOL, P, SIMULATION_LENGTH
)


_VALID_ICM_PARAMS = {AND: {0.80, 0.85, 0.90, 0.95, -1}, OR: {0.05, 0.10, 0.15, 0.20, -1}}


@dataclass(frozen=True)
class MLNInfo:
    """Base class to keep network with its ground truth data and prediciton params."""

    mln_type: str
    mln_name: str
    mln: MultilayerNetwork
    icm_protocol: Literal["OR", "AND"]
    icm_p: float
    x_type: Literal["zeros", "scrapped", "centralities"]
    y_type: list[Literal["exposed", "simulation_length", "peak_iteration", "peak_infected"]]
    sp_raw: DataFrame
    mln_torch: MultilayerNetworkTorch | None = None

    def __post_init__(self) -> None:
        assert self.icm_protocol in {OR, AND}
        assert {self.icm_p}.issubset(_VALID_ICM_PARAMS[self.icm_protocol])
        assert self.x_type in {"zeros", "scrapped", "centralities"}
        assert set(self.y_type).issubset({EXPOSED, SIMULATION_LENGTH, PEAK_ITERATION, PEAK_INFECTED})
        object.__setattr__(self, "mln", functions.remove_selfloop_edges(self.mln))
        object.__setattr__(self, "mln_torch", MultilayerNetworkTorch.from_mln(self.mln))

    def transform_sp(self) -> DataFrame:
        """Transform DataFrame with raw spreading potentials according to given parameters."""
        spreading_potential_columns = [ACTOR, PROTOCOL, P, *self.y_type]
        df = self.sp_raw[spreading_potential_columns]
        df = df[df[PROTOCOL] == self.icm_protocol].drop(PROTOCOL, axis=1)
        if self.icm_p == -1:
            valid_p = _VALID_ICM_PARAMS[self.icm_protocol].symmetric_difference({-1})
            df = df[df[P].isin(valid_p)].drop(P, axis=1)
        else:
            df = df[df[P] == self.icm_p].drop(P, axis=1)
        df = df.groupby([ACTOR]).mean().reset_index()
        return df

    @property
    def name(self) -> str:
        return f"{self.mln_type}_{self.mln_name}"
