import torch
import torch.nn as nn

DEFAULT_EPS = 1e-10
PADDED_Y_VALUE=-1

class SListMLELoss(nn.Module):
    def __init__(
        self,
        w_sl: float,
        w_e: float,
        w_pin: float,
        w_pit: float,
    ) -> None:
        super().__init__()

        self.w_sl = w_sl
        self.w_e = w_e
        self.w_pin = w_pin
        self.w_pit = w_pit

    def forward(
        self,
        pred_values: torch.Tensor,
        true_values: torch.Tensor,
    ) -> torch.Tensor:
        scores_pred = (
            (1 - pred_values[:, 0]) * self.w_sl +
            pred_values[:, 0] * self.w_e +
            (1 - pred_values[:, 0]) * self.w_pit +
            pred_values[:, 0] * self.w_pin
        ).squeeze(-1).unsqueeze(0)

        scores_true = (
            (1 - true_values[:, 0]) * self.w_sl +
            true_values[:, 0] * self.w_e +
            (1 - true_values[:, 0]) * self.w_pit +
            true_values[:, 0] * self.w_pin
        ).squeeze(-1).unsqueeze(0)

        # sorted_indices = torch.argsort(scores_true, dim=1, descending=True)
        # sorted_pred = torch.gather(scores_pred, dim=1, index=sorted_indices)

        # cumsum = torch.logcumsumexp(sorted_pred, dim=1)
        # loss = (cumsum - sorted_pred).sum(dim=1)
        # loss = loss.mean()
        
        loss = self.listMLE(scores_pred, scores_true)
        return loss

    def listMLE(
        self,
        y_pred: torch.Tensor,
        y_true: torch.Tensor,
        eps: float = DEFAULT_EPS,
        padded_value_indicator: int = PADDED_Y_VALUE,
    ) -> torch.Tensor:
        """
        ListMLE loss introduced in "Listwise Approach to Learning to Rank - Theory and Algorithm".
        :param y_pred: predictions from the model, shape [batch_size, slate_length]
        :param y_true: ground truth labels, shape [batch_size, slate_length]
        :param eps: epsilon value, used for numerical stability
        :param padded_value_indicator: an indicator of the y_true index containing a padded item, e.g. -1
        :return: loss value, a torch.Tensor
        """
        # shuffle for randomised tie resolution
        random_indices = torch.randperm(y_pred.shape[-1])
        y_pred_shuffled = y_pred[:, random_indices]
        y_true_shuffled = y_true[:, random_indices]

        y_true_sorted, indices = y_true_shuffled.sort(descending=True, dim=-1)

        mask = y_true_sorted == padded_value_indicator

        preds_sorted_by_true = torch.gather(y_pred_shuffled, dim=1, index=indices)
        preds_sorted_by_true[mask] = float("-inf")

        max_pred_values, _ = preds_sorted_by_true.max(dim=1, keepdim=True)

        preds_sorted_by_true_minus_max = preds_sorted_by_true - max_pred_values

        cumsums = torch.cumsum(preds_sorted_by_true_minus_max.exp().flip(dims=[1]), dim=1).flip(dims=[1])

        observation_loss = torch.log(cumsums + eps) - preds_sorted_by_true_minus_max

        observation_loss[mask] = 0.0

        return torch.mean(torch.sum(observation_loss, dim=1))
