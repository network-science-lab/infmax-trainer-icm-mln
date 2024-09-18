from torch.nn import Module

class BaseHeteroModule(Module):
    def __init__(self, is_hetero: bool,) -> None:
        super().__init__()
        self.is_hetero = is_hetero
        