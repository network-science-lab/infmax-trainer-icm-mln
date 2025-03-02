"""A script with various classess to aggregate layerwise embeddings."""

import torch
import torch.nn.functional as F


class MaxAggregation(torch.nn.Module):
    """Auxuilary class for max aggregation of mln-layers embeddings."""

    def forward(self, h: dict[str, torch.Tensor]) -> torch.Tensor:
        """
        Maximum aggregation of mln layers' embeddings.

        :param h: mln layers' embeddings dict `{nb_mln_layers: [hidden_dim, nb_mln_actors]}`
        :return: a tensor of shape `[hidden_dim, nb_mln_actors]`
        """
        h = torch.stack(list(h.values()))
        return torch.amax(h, dim=0)


class MinAggregation(torch.nn.Module):
    """Auxuilary class for max aggregation of mln-layers embeddings."""

    def forward(self, h: dict[str, torch.Tensor]) -> torch.Tensor:
        """
        Minimum aggregation of mln layers' embeddings.

        :param h: mln layers' embeddings dict `{nb_mln_layers: [hidden_dim, nb_mln_actors]}`
        :return: a tensor of shape `[hidden_dim, nb_mln_actors]`
        """
        h = torch.stack(list(h.values()))
        return torch.amin(h, dim=0)


class AvgAggregation(torch.nn.Module):
    """Auxuilary class for average aggregation of mln-layers embeddings."""

    def forward(self, h: dict[str, torch.Tensor]) -> torch.Tensor:
        """
        Average aggregation of mln layers' embeddings.

        :param x: mln layers' embeddings dict `{nb_mln_layers: [hidden_dim, nb_mln_actors]}`
        :return: a tensor of shape `[hidden_dim, nb_mln_actors]`
        """
        h = torch.stack(list(h.values()))
        return torch.mean(h, dim=0)


class SumAggregation(torch.nn.Module):
    """Auxuilary class for summation aggregation of mln-layers embeddings."""

    def forward(self, h: dict[str, torch.Tensor]) -> torch.Tensor:
        """
        Summation aggregation of mln layers' embeddings.

        :param h: mln layers' embeddings dict `{nb_mln_layers: [hidden_dim, nb_mln_actors]}`
        :return: a tensor of shape `[hidden_dim, nb_mln_actors]`
        """
        h = torch.stack(list(h.values()))
        return torch.sum(h, dim=0)


class LayerwiseAggregation(torch.nn.Module):
    """Auxuilary class for trainable custom aggregation of mln-layers embeddings."""

    def __init__(self, hidden_channels: int) -> None:
        """Initialise the object."""
        super().__init__()
        self.attn = torch.nn.Linear(hidden_channels, 1)

    def forward(self, h: dict[str, torch.Tensor]) -> torch.Tensor:
        """
        Trainable aggregation of mln layers' embeddings.

        :param h: mln layers' embeddings dict `{nb_mln_layers: [hidden_dim, nb_mln_actors]}`
        :return: a tensor of shape `[hidden_dim, nb_mln_actors]`
        """
        h = torch.stack(list(h.values()))
        attn_scores = self.attn(h)
        attn_scores = F.softmax(attn_scores, dim=0)
        weighted_sum = (attn_scores * h).sum(dim=0)
        return weighted_sum


class AttentionAggregation(torch.nn.Module):
    """Auxuilary class for trainable attention aggregation of mln-layers embeddings."""

    def __init__(self, hidden_channels: int) -> None:
        """Initialise the object."""
        super().__init__()
        self.V = torch.nn.Bilinear(hidden_channels, hidden_channels, 1, bias=False)
        self.q = torch.nn.Sequential(
            torch.nn.Linear(hidden_channels, hidden_channels),
            # torch.nn.LeakyReLU(),
            torch.nn.ReLU(),
        )
        self.reset_parameters()

    def reset_parameters(self) -> None:
        """Reset parameters of the trainable modules."""
        self.V.reset_parameters()
        self.q[0].reset_parameters()

    def forward(self, h: dict[str, torch.Tensor]) -> torch.Tensor:
        """
        Trainable aggregation of mln layers' embeddings.

        :param h: mln layers' embeddings dict `{nb_mln_layers: [hidden_dim, nb_mln_actors]}`
        :return: a tensor of shape `[hidden_dim, nb_mln_actors]`
        """
        b = torch.cat(
            [
                self.V(self.q(h[edge_type]), h[edge_type]).tanh()
                for edge_type in h.keys()
            ],
            dim=-1,
        )
        # alpha = b.softmax(dim=0)
        alpha = b.softmax(dim=-1)
        embeddings = torch.stack(self._to_ordered(h), dim=-1)
        # embeddings = torch.stack(list(h.values()))
        
        z = (alpha.unsqueeze(dim=1) * embeddings).sum(dim=-1)
        return z
        # return (alpha.unsqueeze(dim=0).permute(2, 1, 0) * embeddings).sum(dim=0)
    
    def _to_ordered(self, h: dict[str, torch.Tensor]) -> list[torch.Tensor]:
        return [h[et] for et in h.keys()]


class SelfAttentionLayer_(torch.nn.Module):
    def __init__(self, in_channels, out_channels, num_heads=4):
        super(SelfAttentionLayer, self).__init__()
        self.attention = torch.nn.MultiheadAttention(embed_dim=out_channels, num_heads=num_heads, batch_first=True)
        self.norm = torch.nn.LayerNorm(out_channels)  # Normalizacja po atencji
        self.proj = torch.nn.Linear(in_channels, out_channels) if in_channels != out_channels else torch.nn.Identity()
    
    def forward(self, x):
        x = self.proj(x)  # Dopasowanie wymiarów, jeśli konieczne
        x_residual = x  # Skip connection
        x, _ = self.attention(x, x, x)  # Self-attention (Q=K=V)
        x = self.norm(x + x_residual)  # Dodajemy skip-connection + normalizujemy
        return x

class SelfAttentionLayer(torch.nn.Module):
    def __init__(self, in_channels, out_channels=None):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels or in_channels  # Domyślnie pozostaje taki sam

        self.key = torch.nn.Linear(in_channels, self.out_channels)
        self.query = torch.nn.Linear(in_channels, self.out_channels)
        self.value = torch.nn.Linear(in_channels, self.out_channels)

        # Jeśli out_channels różni się od in_channels, dodaj liniową transformację na wyjściu
        self.output_proj = (
            torch.nn.Linear(self.out_channels, out_channels) if out_channels and out_channels != in_channels else torch.nn.Identity()
        )

    def forward(self, x, mask=None):
        keys = self.key(x)
        queries = self.query(x)
        values = self.value(x)

        scores = torch.matmul(queries, keys.transpose(-2, -1)) / torch.sqrt(torch.tensor(self.out_channels, dtype=torch.float32))

        if mask is not None:
            scores = scores.masked_fill(mask == 0, -1e9)

        attention_weights = F.softmax(scores, dim=-1)
        output = torch.matmul(attention_weights, values)
        output = self.output_proj(output)
        return output