"""Transformations of the MLNHeterodata to improve learning."""

import torch
from torch_geometric.transforms import BaseTransform, Compose

from _data_set.nsl_data_utils.loaders.constants import ACTOR
from src.data_models.mln_hetero_data import MLNHeteroData


def plot_distr(data:MLNHeteroData, title: str):
    import matplotlib.pyplot as plt; import pandas as pd
    pd.DataFrame(data[ACTOR].y.numpy()).hist()
    plt.savefig(title, dpi=300, bbox_inches="tight")
    plt.close()


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

    def __call__(self, data: MLNHeteroData) -> MLNHeteroData:
        data[ACTOR].x = data[ACTOR].x / data[ACTOR].x.max(dim=0).values
        data[ACTOR].y = data[ACTOR].y / data[ACTOR].y.max(dim=0).values
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
