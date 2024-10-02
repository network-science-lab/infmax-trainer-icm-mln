import torch


class WeightedMSE(torch.nn.Module):
    def __init__(
        self,
        w_sl: float,
        w_e: float,
        w_pin: float,
        w_pit: float,
        reduction: str,
    ) -> None:
        super().__init__()
        self.weights = torch.tensor([w_sl, w_e, w_pin, w_pit], dtype=torch.float32)
        self.mse = torch.nn.MSELoss(reduction=reduction)

    def forward(self, y_hat: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        loss = []
        for i, weight in enumerate(self.weights):
            loss_i = weight * self.mse(y_hat[:, i], y[:, i])
            loss.append(loss_i)

        return sum(loss)
