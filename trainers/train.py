"""
load dataset
load model
train model
evaluate model
"""
from dataclasses import dataclass
import network_diffusion as nd

from misc.net_loader import load_network
from ss_models.loader import load_model


@dataclass(frozen=True)
class Network:
    name: str
    graph: nd.MultilayerNetwork | nd.MultilayerNetworkTorch


def train(args):
    networks = [Network(n, load_network(net_name=n, as_tensor=True)) for n in args["networks"]]
    model = load_model(args["model"])
    for network in networks:
        print(network.name)
        model(network=network.graph)

    