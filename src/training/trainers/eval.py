"""Functions to help in evaluating performance of a given seed set."""

from typing import Any, Literal

import network_diffusion as nd
import pandas as pd

from _data_set.nsl_data_utils.loaders.net_loader import load_network
from _data_set.nsl_data_utils.loaders.sp_loader import get_gt_data
from _data_set.nsl_data_utils.models.torch_model import (TorchMICModel,
                                                         TorchMICSimulator)


def evaluate_seed_set(
    net: nd.MultilayerNetworkTorch,
    seed_set: list[Any],
    protocol: Literal["OR", "AND"],
    probability: float,
    n_steps: int,
    n_repetitions: int,
) -> pd.DataFrame:
    """Run multilayer ICM on given seed set and model's parameters."""
    results = []
    for _ in range(n_repetitions):
        micm = TorchMICModel(protocol=protocol, probability=probability)
        simulator = TorchMICSimulator(
            model=micm, net=net, n_steps=n_steps, seed_set=seed_set
        )
        result = simulator.perform_propagation()
        results.append(result)
    return pd.DataFrame(results)


if __name__ == "__main__":
    from _data_set.infmax_data_utils.loaders.constants import *

    net_name = TOY_NETWORK
    proto = OR
    p = 0.25
    n_steps = 10000
    budget = 5
    n_repetitions = 30

    net = load_network(net_name, as_tensor=True)
    seed_set = get_gt_data(net_name, proto, p, budget)
    raw_results = evaluate_seed_set(net, seed_set, proto, p, n_steps, n_repetitions)
    print("Performance of given seed set:\n", raw_results.mean())
