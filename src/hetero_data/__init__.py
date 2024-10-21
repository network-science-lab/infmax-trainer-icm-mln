from network_diffusion.mln.actor import MLNetworkActor
from network_diffusion.mln.functions import (betweenness, closeness,
                                             core_number, degree,
                                             neighbourhood_size,
                                             voterank_actorwise)
from network_diffusion.mln.mlnetwork import MultilayerNetwork


def voterank(net: MultilayerNetwork) -> dict[MLNetworkActor, int]:
    actors = voterank_actorwise(net)
    return {actor: idx for idx, actor in enumerate(actors[::-1])}


CENTRALITY_FUNCTIONS = [
    degree,
    betweenness,
    closeness,
    core_number, # related with k-shell-mln
    neighbourhood_size,
    voterank,
]
