"""
load dataset
load model
train model
evaluate model
"""
from dataclasses import dataclass
import network_diffusion as nd

from misc.net_loader import load_network
from infmax_models.loader import load_model


@dataclass(frozen=True)
class Network:
    name: str
    graph: nd.MultilayerNetwork | nd.MultilayerNetworkTorch


def train(args):
    networks = [Network(n, load_network(net_name=n, as_tensor=True)) for n in args["networks"]]
    model = load_model(model_config=args["model"], train_config=args["train"])
    for network in networks:
        print(f"Dataset: {network.name}")
        seeds = model(network=network.graph)
        print(f"Chosen following actors as seeds: {seeds}\n")

    