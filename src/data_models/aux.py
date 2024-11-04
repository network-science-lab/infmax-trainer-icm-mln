import network_diffusion as nd


def voterank(net: nd.mln.MultilayerNetwork) -> dict[nd.mln.MLNetworkActor, int]:
    actors = nd.mln.functions.voterank_actorwise(net)
    return {actor: idx for idx, actor in enumerate(actors[::-1])}


CENTRALITY_FUNCTIONS = [
    nd.mln.functions.degree,
    nd.mln.functions.betweenness,
    nd.mln.functions.closeness,
    nd.mln.functions.core_number, # related with k-shell-mln
    nd.mln.functions.neighbourhood_size,
    voterank,
]
