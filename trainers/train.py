"""
load dataset
load model
train model
evaluate model
"""
import network_diffusion as nd

from dataclasses import dataclass

from misc.net_loader import load_network
from misc.sp_loader import load_sp
from infmax_models.loader import load_model


@dataclass(frozen=True)
class Network:
    name: str
    graph: nd.MultilayerNetwork | nd.MultilayerNetworkTorch

# TODO: for now it's just a mock. we have to implement a real training pipeline

def train(args):
    for n in args["networks"]:
        print(n)
        sp = load_sp(net_name=n, mean_data=True)
        print(sp.head())

    # networks = [Network(n, load_network(net_name=n, as_tensor=True)) for n in args["networks"]]
    # model = load_model(model_config=args["model"], train_config=args["train"])
    # for network in networks:
    #     print(f"Dataset: {network.name}")
    #     seeds = model(network=network.graph)
    #     print(f"Chosen following actors as seeds: {seeds}\n")

    