import torch
from torch.nn import Linear, Module, MultiheadAttention


class CrossRelationAttention(Module):

    def __init__(self, nb_hidden_channels: int, pooling_factor: int):
        super().__init__()

        # Linear projections for queries, keys, and values
        self.query_proj = Linear(nb_hidden_channels, nb_hidden_channels)
        self.key_proj = Linear(nb_hidden_channels, nb_hidden_channels)
        self.value_proj = Linear(nb_hidden_channels, nb_hidden_channels)

        # Multi-head attention layer
        self.attn = MultiheadAttention(embed_dim=nb_hidden_channels, num_heads=1)

        # Shrink output dimention
        self.pool = Linear(nb_hidden_channels, nb_hidden_channels // pooling_factor)

    def forward(self, relation_embeddings):
        """
        :param relation_embeddings: List of tensors for each relation, each shaped (nb_agents, nb_hidden_channels)
        :return: List of updated relation embeddings
        """
        nb_relations = len(relation_embeddings)

        # Stack all relation embeddings for multi-head attention
        stacked_embeddings = torch.stack(relation_embeddings)  # (nb_relations, nb_agents, nb_hidden_channels)

        # Prepare query, key, and value from the relation embeddings
        queries = self.query_proj(stacked_embeddings)  # (nb_relations, nb_agents, nb_hidden_channels)
        keys = self.key_proj(stacked_embeddings)  # (nb_relations, nb_agents, nb_hidden_channels)
        values = self.value_proj(stacked_embeddings)  # (nb_relations, nb_agents, nb_hidden_channels)

        # Transpose for MultiheadAttention format (seq_len, batch_size, embed_dim)
        queries = queries.permute(1, 0, 2)  # (nb_agents, nb_relations, nb_hidden_channels)
        keys = keys.permute(1, 0, 2)  # (nb_agents, nb_relations, nb_hidden_channels)
        values = values.permute(1, 0, 2)  # (nb_agents, nb_relations, nb_hidden_channels)

        # Perform multi-head attention and permute back to original shape
        attn_output, _ = self.attn(queries, keys, values)  # (nb_agents, nb_relations, nb_hidden_channels)
        attn_output = attn_output.permute(1, 0, 2)

        # Perform pooling
        attn_output = self.pool(attn_output)

        # Return the updated embeddings
        return [attn_output[relation_idx, ...] for relation_idx in range(nb_relations)]  # (nb_relations, nb_agents, nb_hidden_channels)
