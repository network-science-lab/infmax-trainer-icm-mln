"""Loader for ground truth dataset on spreading potentials."""

from functools import wraps
from pathlib import Path
from typing import Callable

import pandas as pd
from misc.constants import *
from _data_set.eda_utils.globals import *
from _data_set.eda_utils.csv_loader import read_csv


def _get_sp(csv_paths: list[str]) -> pd.DataFrame:
    """Read spreading potentials stored in files indicated by given regex."""
    raw_csvs = []
    for _, file_path in enumerate(csv_paths):
        # print(f"processing {_}th file: {file_path.name}")
        raw_csvs.append(read_csv(file_path))
    sp = pd.concat(raw_csvs, axis=0, ignore_index=True)
    assert len(sp["network"].unique()) == 1
    sp[[SIMULATION_LENGTH, EXPOSED, NOT_EXPOSED, PEAK_INFECTED, PEAK_ITERATION, P]] = sp[
        [SIMULATION_LENGTH, EXPOSED, NOT_EXPOSED, PEAK_INFECTED, PEAK_ITERATION, P]
    ].apply(pd.to_numeric)
    return sp


def _get_csv_paths(csv_regex: str) -> list[str]:
    return list(Path(".").glob(csv_regex))


def _sp_not_implemented():
    raise NotImplementedError(f"Spreading potentials have been not prepared yet!")


def mean_sp_data(load_sp_func: Callable) -> Callable:
    """Decorate loader of spreading potentials to mean them on the fly."""
    @wraps(load_sp_func)
    def wrapper(*args, mean_data: bool, **kwargs) -> pd.DataFrame:
        sp_raw = load_sp_func(*args, **kwargs)
        if mean_data:
            sp_grouped  = sp_raw.groupby(by=[NETWORK, PROTOCOL, ACTOR])
            sp_mean = sp_grouped.mean()[
                [SIMULATION_LENGTH, EXPOSED, NOT_EXPOSED, PEAK_INFECTED, PEAK_ITERATION]
            ]
            return sp_mean
        return sp_raw
    return wrapper


@mean_sp_data
def load_sp(net_name: str) -> pd.DataFrame:
    """Load spreading potentials dataset for given network."""
    if net_name == FMRI74:
        _sp_not_implemented()
    elif net_name == ARXIV_NETSCIENCE_COAUTHORSHIP:
        csv_paths = [
            f for f in Path(f"{SP_PREFIX}/arxiv_netscience_coauthorship").rglob('**/*.csv') 
            if 'math.oc' not in f.parts
        ]
        return _get_sp(csv_paths)
    elif net_name == ARXIV_NETSCIENCE_COAUTHORSHIP_MATH:
        return _get_sp(_get_csv_paths(f"{SP_PREFIX}/arxiv_netscience_coauthorship/math.oc/*.csv"))
    elif net_name == AUCS:
        return _get_sp(_get_csv_paths(f"{SP_PREFIX}/small_real/*--net-aucs.csv"))
    elif net_name == CANNES:
        _sp_not_implemented()
    elif net_name == CKM_PHYSICIANS:
        return _get_sp(_get_csv_paths(f"{SP_PREFIX}/small_real/*--net-ckm_physicians.csv"))
    elif net_name == EU_TRANSPORTATION:
        return _get_sp(_get_csv_paths(f"{SP_PREFIX}/small_real/*--net-eu_transportation.csv"))
    elif net_name == EU_TRANSPORT_KLM:
        return _get_sp(_get_csv_paths(f"{SP_PREFIX}/small_real/*--net-eu_transport_klm.csv"))
    elif net_name == LAZEGA:
        return _get_sp(_get_csv_paths(f"{SP_PREFIX}/small_real/*--net-lazega.csv"))
    elif net_name == ER1:
        _sp_not_implemented()
    elif net_name == ER2:
        _sp_not_implemented()
    elif net_name == ER3:
        _sp_not_implemented()
    elif net_name == ER5:
        _sp_not_implemented()
    elif net_name == SF1:
        _sp_not_implemented()
    elif net_name == SF2:
        _sp_not_implemented()
    elif net_name == SF3:
        _sp_not_implemented()
    elif net_name == SF5:
        _sp_not_implemented()
    elif net_name == TIMIK1Q2009:
        return _get_sp(_get_csv_paths(f"{SP_PREFIX}/timik1q2009/**/*.csv"))
    elif net_name == TOY_NETWORK:
        return _get_sp(_get_csv_paths(f"{SP_PREFIX}/small_real/*--net-toy_network.csv"))
    raise AttributeError(f"Unknown network: {net_name}")
