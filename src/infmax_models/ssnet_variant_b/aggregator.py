import torch
import torch.nn.functional as F


class LayerwiseAggregation(torch.nn.Module):
    """Auxuilary class for trainable aggregation of mln-layers embeddings."""

    def __init__(self, hidden_channels: int, output_dim: int) -> None:
        """Initialise the object."""
        super().__init__()
        self.attn = torch.nn.Linear(hidden_channels, 1)
        self.head = torch.nn.Sequential(
            torch.nn.Linear(in_features=hidden_channels, out_features=output_dim),
            torch.nn.Softplus(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Trainable aggregation of mln layers' embeddings.

        :param x: a tensor with mln layers' embeddings of shape `[nb_mln_layers, hidden_dim, nb_mln_actors]`
        :return: a tensor with aggregated x of shape `[1, hidden_dim, nb_mln_actors]`
        """
        attn_scores = self.attn(x)
        attn_scores = F.softmax(attn_scores, dim=0)
        weighted_sum = (attn_scores * x).sum(dim=0)
        return self.head(weighted_sum)
