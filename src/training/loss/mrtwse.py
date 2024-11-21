"""Implementation of the cusrom Mean Root Total Weighted Squared Error."""

import torch


class MRTWSE(torch.nn.Module):
    def __init__(
        self,
        w_sl: float,
        w_e: float,
        w_pin: float,
        w_pit: float,
        use_abs_negative_penalty: bool = False,
    ) -> None:
        """
        Mean Root Total Weighted Squared Error.

        1. scale weights by softmax so that they sum to 1
        2. compute squared error
        3. scale it by weights
        4. for each actor sum the vector of partial errors and obtain squared root of it
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
        self._penalty = use_abs_negative_penalty

    def forward(self, y_hat: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        if not self._bypass_flag:
            self.weights = self.weights.to(device=y.device)
            self._bypass_flag = True
        se = (y_hat - y) ** 2  # squared error
        wse = se * self.weights.expand_as(se)  # weighted se
        rtwse = torch.sqrt(
            wse.sum(dim=1)
        )  # root total wse (for each y_i sum err and sqrt them)
        mrtwse = rtwse.mean()  # compute mean rtwse in the batch

        if self._penalty:
            negative_distance = y_hat[y_hat < 0].abs()
            penalty = negative_distance.mean()
            if not torch.isnan(penalty) or penalty > 0:
                mrtwse += penalty.detach()

        return mrtwse
