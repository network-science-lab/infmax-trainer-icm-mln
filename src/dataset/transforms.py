"""Transformations of the MLNHeterodata to improve learning."""

import torch
from torch_geometric.transforms import BaseTransform, Compose

from _data_set.nsl_data_utils.loaders.constants import ACTOR
from src.data_models.mln_hetero_data import MLNHeteroData
import networkx as nx
from _data_set.nsl_data_utils.loaders.net_loader import load_network


def plot_distr(data:MLNHeteroData, title: str):
    import matplotlib.pyplot as plt; import pandas as pd
    pd.DataFrame(data[ACTOR].y.numpy()).hist()
    plt.savefig(title, dpi=300, bbox_inches="tight")
    plt.close()


class NormaliseByDomain(BaseTransform): # TODO:
    """Default transformation used in experiments."""

    def __init__(self) -> None:
        super().__init__()
    
    @staticmethod
    def _get_norm_matrix(x: torch.Tensor) -> torch.Tensor:
        norm_max = x.max(dim=0).values
        norm_max[norm_max == 0.] = 1.
        return norm_max

    def __call__(self, data: MLNHeteroData) -> MLNHeteroData:
        mln = load_network(
            data.network_type,
            data.network_name,
            as_tensor=False,
        )
        
        diameter = 0 # TODO: CLEAN DIRTY CODE
        for layer in mln.layers:
            try:
                diameter += nx.diameter(mln[layer])
            except:
                largest_cc = max(nx.connected_components(mln[layer]), key=len)
                g_largest_cc = mln[layer].subgraph(largest_cc).copy()
                diameter += nx.diameter(g_largest_cc)
        diameter = diameter / len(mln.layers)
        del mln
        
        data[ACTOR].y[:,0] = data[ACTOR].y[:,0] / diameter
        data[ACTOR].y[:,1] = data[ACTOR].y[:,1] / len(data.actors_map)
        data[ACTOR].y[:,2] = data[ACTOR].y[:,2] / diameter
        data[ACTOR].y[:,3] = data[ACTOR].y[:,3] / len(data.actors_map)
        return data


class NormaliseByActorsNumber(BaseTransform):
    """Default transformation used in experiments."""

    def __init__(self) -> None:
        super().__init__()

    def __call__(self, data: MLNHeteroData) -> MLNHeteroData:
        data[ACTOR].x = data[ACTOR].x / len(data.actors_map)
        data[ACTOR].y = data[ACTOR].y / len(data.actors_map)
        # plot_distr(data, "distribution_normact.png")
        return data


class NormaliseByMax(BaseTransform):
    """Default transformation used in experiments."""

    def __init__(self) -> None:
        super().__init__()
    
    @staticmethod
    def _get_norm_matrix(x: torch.Tensor) -> torch.Tensor:
        norm_max = x.max(dim=0).values
        norm_max[norm_max == 0.] = 1.
        return norm_max

    def __call__(self, data: MLNHeteroData) -> MLNHeteroData:
        data[ACTOR].x = data[ACTOR].x / self._get_norm_matrix(data[ACTOR].x)
        data[ACTOR].y = data[ACTOR].y / self._get_norm_matrix(data[ACTOR].y)
        # plot_distr(data, "distribution_normmax.png")
        return data


class ScatterWithExponent(BaseTransform):
    """Apply if data is too concentrated, keeps data in the original range."""

    def __init__(self, scatter_factor: float) -> None:
        super().__init__()
        if scatter_factor <= 0:
            raise AttributeError("scatter factor should be >= 0")
        self.sf = scatter_factor

    def __call__(self, data: MLNHeteroData) -> MLNHeteroData:
        data[ACTOR].y = torch.exp(self.sf * data[ACTOR].y) / torch.exp(torch.tensor(self.sf))
        # plot_distr(data, "distribution_exp.png")
        return data


class ConctractWithLog(BaseTransform):
    """Apply to narrow down data."""

    def __init__(self) -> None:
        super().__init__()

    def __call__(self, data: MLNHeteroData) -> MLNHeteroData:
        data[ACTOR].y = torch.log(data[ACTOR].y + 1)
        # plot_distr(data, "distribution_log.png")
        return data


class NormaliseAndScatter(BaseTransform):
    """NormaliseByMax -> ScatterWithExponent"""

    def __init__(self, scatter_factor: float) -> None:
        super().__init__()
        self.transform = Compose([NormaliseByMax(), ScatterWithExponent(scatter_factor)])
    
    def __call__(self, data: MLNHeteroData) -> MLNHeteroData:
        # plot_distr(data, "distribution_raw.png")
        return self.transform(data)


class ConctractAndNormalise(BaseTransform):
    """ScatterWithExponent -> NormaliseByMax"""

    def __init__(self) -> None:
        super().__init__()
        self.transform = Compose([ConctractWithLog(), NormaliseByMax()])

    def __call__(self, data: MLNHeteroData) -> MLNHeteroData:
        # plot_distr(data, "distribution_raw.png")
        return self.transform(data)
