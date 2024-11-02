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
        self.weights = torch.tensor([w_sl, w_e, w_pin, w_pit], dtype=torch.float32)  # TODO: here is a bug - device is CPU!
        self.mse = torch.nn.MSELoss(reduction=reduction)

    def forward(self, y_hat: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        loss = 0
        for i, weight in enumerate(self.weights):
            loss += weight * self.mse(y_hat[:, i], y[:, i])

        return loss
