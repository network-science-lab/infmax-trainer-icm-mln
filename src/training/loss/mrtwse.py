"""Implementation of the cusrom Mean Root Total Weighted Squared Error."""

import torch


class MTWAE(torch.nn.Module):
    def __init__(
        self,
        w_sl: float,
        w_e: float,
        w_pin: float,
        w_pit: float,
    ) -> None:
        """
        Mean Total Weighted Absolute Error.

        1. scale weights by softmax so that they sum to 1
        2. compute absolute error
        3. scale it by weights
        4. for each actor sum the vector of partial errors
        5. return mean value of such errors along the batch

        :param w_sl: weight for simulation length
        :param w_e: weight for exposed number
        :param w_pin: weight for peak infection
        :param w_pit: weight for peak iteration
        """
        super().__init__()
        raw_weights = torch.tensor([w_sl, w_e, w_pin, w_pit], dtype=torch.float32)
        self.weights = torch.softmax(raw_weights, dim=0)
        self._bypass_flag = False

    def forward(self, y_hat: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        if not self._bypass_flag:
            self.weights = self.weights.to(device=y.device)
            self._bypass_flag = True
        ae = torch.abs(y_hat - y)  # absolute error
        wae = ae * self.weights.expand_as(ae)  # weighted ae
        twae = wae.sum(dim=1)  # total wae
        mtwae = twae.mean()  # compute mean swae in the batch
        return mtwae
