"""
load dataset
load model
train model
evaluate model
"""
import argparse
from dataclasses import dataclass

import network_diffusion as nd
from _data_set.nsl_data_utils.loaders.net_loader import load_network
from _data_set.nsl_data_utils.loaders.sp_loader import get_gt_data
from infmax_models.loader import load_model
from trainers.eval import evaluate_seed_set


@dataclass(frozen=True)
class Network:
    name: str
    graph: nd.MultilayerNetwork | nd.MultilayerNetworkTorch

# TODO: for now it's just a mock. we have to implement a real training pipeline

def train(args: argparse.Namespace) -> None:

    # load dataset
    networks = [Network(n, load_network(net_name=n, as_tensor=True)) for n in args["networks"]]

    # load model
    model = load_model(model_config=args["model"], train_config=args["train"])

    # capture parameters of spreading regime
    proto = args["spreading_regime"]["protocol"]
    p = args["spreading_regime"]["p"]
    n_steps = args["spreading_regime"]["n_steps"]
    n_repetitions = args["spreading_regime"]["n_repetitions"]
    seed_size = args["train"]["seed_size"]

    for net in networks:
        print(f"Dataset: {net.name}")

        pred_seeds = model(network=net.graph)
        pred_performance = evaluate_seed_set(
            net=net.graph,
            seed_set=pred_seeds,
            protocol=proto,
            probability=p,
            n_steps=n_steps,
            n_repetitions=n_repetitions,
        )
        print(f"Predicted seed set: {pred_seeds}")
        print(f"{pred_performance.mean()}\n")

        ref_seeds = get_gt_data(net.name, proto, p, seed_size)
        ref_performance = evaluate_seed_set(
            net=net.graph,
            seed_set=ref_seeds,
            protocol=proto,
            probability=p,
            n_steps=n_steps,
            n_repetitions=n_repetitions
        )
        print(f"Reference seed set: {ref_seeds}")
        print(f"{ref_performance.mean()}\n")
