import torch


class WeightedMSE(torch.nn.Module):
    def __init__(
        self,
        w_sl: float,
        w_e: float,
        w_pin: float,
        w_pit: float,
        reduction: str = "mean",
    ) -> None:
        super().__init__()
        self.weights = torch.tensor([w_sl, w_e, w_pin, w_pit], dtype=torch.float32)
        self.mse = torch.nn.MSELoss(reduction=reduction)
        self._bypass_flag = False

    def forward(self, y_hat: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        if not self._bypass_flag:
            self.weights = self.weights.to(device=y.device)
            self._bypass_flag = True

        mse = self.weights * self.mse(y_hat, y)
        loss = sum(mse)

        return loss
