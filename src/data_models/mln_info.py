"""A script where auxiliary data model is implemented."""

import re
from dataclasses import dataclass
from pathlib import Path
from typing import Literal

from _data_set.nsl_data_utils.loaders.centrality_loader import load_centralities_path
from _data_set.nsl_data_utils.loaders.constants import (
    AND,
    EXPOSED,
    OR,
    PEAK_INFECTED,
    PEAK_ITERATION,
    SIMULATION_LENGTH,
)
from _data_set.nsl_data_utils.loaders.sp_loader import load_sp_paths

_VALID_ICM_PARAMS = {
    AND: {0.80, 0.85, 0.90, 0.95, -1},
    OR: {0.05, 0.10, 0.15, 0.20, -1},
}


@dataclass(frozen=True)
class MLNInfo:
    """Base class to keep paths and params required to load a network and GT."""

    mln_type: str
    mln_name: str
    icm_protocol: Literal["OR", "AND"]
    icm_p: float
    x_type: Literal["zeros", "scrapped", "centralities"]
    y_type: list[
        Literal["exposed", "simulation_length", "peak_iteration", "peak_infected"]
    ]
    sp_paths: list[Path]
    ft_path: Path

    @property
    def name(self) -> str:
        return f"{self.mln_type}_{self.mln_name}"

    @staticmethod
    def _validate_args(
        icm_protocol: str,
        icm_p: float,
        x_type: str,
        y_type: str,
    ) -> None:
        assert icm_protocol in {OR, AND}
        assert {icm_p}.issubset(_VALID_ICM_PARAMS[icm_protocol])
        assert x_type in {"zeros", "scrapped", "centralities"}
        assert set(y_type).issubset(
            {EXPOSED, SIMULATION_LENGTH, PEAK_ITERATION, PEAK_INFECTED}
        )

    @staticmethod
    def _get_ft_path(
        x_type: str,
        mln_type: str,
        mln_name: str,
    ) -> Path | None:
        if x_type == "zeros":
            return None
        elif x_type == "centralities":
            return load_centralities_path(network_type=mln_type, network_name=mln_name)
        elif x_type == "scrapped":
            raise NotImplementedError(f"{x_type} has not been implemented yet")
        else:
            raise ValueError("Unknown x_type!")

    @staticmethod
    def _filter_sp_path(
        sp_paths: Path,
        icm_protocol: str,
        icm_p: float,
        mln_name: str,
    ) -> list[Path]:
        "Filters filenames based on protocol, p, and the network name."
        if icm_p == -1:
            valid_p_values = _VALID_ICM_PARAMS[icm_protocol].difference({-1})
            regex_vpvs = [r"\.".join(str(vpv).split(".")) for vpv in valid_p_values]
            icm_p_pattern = "|".join(regex_vpvs)
        else:
            icm_p_pattern = re.escape(f"{icm_p:.2f}")
        pattern = rf"^proto-{icm_protocol}--p-({icm_p_pattern})--net-{mln_name}\.csv$"
        return [sp_path for sp_path in sp_paths if re.match(pattern, sp_path.name)]

    @classmethod
    def from_config(
        cls,
        mln_type: str,
        mln_name: str,
        icm_protocol: Literal["OR", "AND"],
        icm_p: float,
        x_type: Literal["zeros", "scrapped", "centralities"],
        y_type: list[
            Literal["exposed", "simulation_length", "peak_iteration", "peak_infected"]
        ],
    ) -> "MLNInfo":
        cls._validate_args(icm_protocol, icm_p, x_type, y_type)
        sp_paths_all = load_sp_paths(net_type=mln_type, net_name=mln_name)
        sp_paths_filtered = cls._filter_sp_path(
            sp_paths_all, icm_protocol, icm_p, mln_name
        )
        ft_path = cls._get_ft_path(x_type, mln_type, mln_name)

        if len(sp_paths_filtered) == 0:
            raise ValueError
        if ft_path is None and x_type == "centralities":
            raise ValueError

        return cls(
            mln_type=mln_type,
            mln_name=mln_name,
            icm_protocol=icm_protocol,
            icm_p=icm_p,
            x_type=x_type,
            y_type=y_type,
            sp_paths=sp_paths_filtered,
            ft_path=ft_path,
        )
