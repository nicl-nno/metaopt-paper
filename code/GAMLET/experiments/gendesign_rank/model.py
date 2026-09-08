"""Identical conditional graph architecture for all loss functions."""

import torch
from torch import nn
from torch.nn import functional as F


class RepositoryGraphSurrogate(nn.Module):
    """GAMLET's actual SimpleGNNEncoder, adapted to mechanical member graphs.

    Both regression and ranking instantiate this exact class. The task MLP and
    scalar head are shared experimental adapters, not the legacy training shell.
    """
    def __init__(self, node_dim=17, context_dim=6, hidden=32, layers=3):
        super().__init__()
        from gamlet.surrogate.encoders.simple_graph_encoder import SimpleGNNEncoder
        self.encoder = SimpleGNNEncoder(in_size=node_dim, d_model=hidden,
                                       gnn_type="graphsage", dropout=0.0,
                                       num_layers=layers, batch_norm=False, in_embed=False)
        self.context = nn.Sequential(nn.Linear(context_dim, 16), nn.ReLU(), nn.Linear(16, 16), nn.ReLU())
        self.head = nn.Sequential(nn.Linear(hidden+16, 32), nn.ReLU(), nn.Linear(32, 1))

    @staticmethod
    def graph_batch(nodes, adjacency, mask):
        from torch_geometric.data import Batch
        active = mask.bool()
        if not bool(active.any(dim=1).all()):
            raise ValueError("Every graph needs an active member")
        # Compact active nodes; absent members must not enter global_mean_pool.
        index = active.long().flatten().cumsum(0).reshape_as(mask)-1
        edges = (adjacency.ne(0) & active[:, :, None] & active[:, None, :]).nonzero()
        edge_index = torch.stack((index[edges[:, 0], edges[:, 1]],
                                  index[edges[:, 0], edges[:, 2]]))
        counts = active.sum(1)
        batch = torch.arange(len(nodes), device=nodes.device).repeat_interleave(counts)
        return Batch(x=nodes[active], edge_index=edge_index, batch=batch,
                     ptr=torch.cat((counts.new_zeros(1), counts.cumsum(0))))

    def forward(self, nodes, adjacency, mask, context):
        pooled = self.encoder(self.graph_batch(nodes, adjacency, mask))
        return self.head(torch.cat((pooled, self.context(context)), dim=-1)).squeeze(-1)


class GraphSurrogate(nn.Module):
    def __init__(self, node_dim=17, context_dim=6, hidden=32, layers=3):
        super().__init__()
        self.input = nn.Linear(node_dim, hidden)
        self.messages = nn.ModuleList([nn.Linear(hidden*2, hidden) for _ in range(layers)])
        self.context = nn.Sequential(nn.Linear(context_dim, 16), nn.ReLU(), nn.Linear(16, 16), nn.ReLU())
        self.head = nn.Sequential(nn.Linear(hidden+16, 32), nn.ReLU(), nn.Linear(32, 1))

    def forward(self, nodes, adjacency, mask, context):
        h = F.relu(self.input(nodes)) * mask.unsqueeze(-1)
        for layer in self.messages:
            neighbors = torch.bmm(adjacency, h)
            h = F.relu(layer(torch.cat([h, neighbors], dim=-1))) * mask.unsqueeze(-1)
        pooled = h.sum(dim=1) / mask.sum(dim=1, keepdim=True).clamp(min=1)
        return self.head(torch.cat([pooled, self.context(context)], dim=-1)).squeeze(-1)


def ranknet_loss(scores, targets):
    """Lower scores are better. Targets are compared only within one task."""
    i, j = torch.triu_indices(len(scores), len(scores), offset=1, device=scores.device)
    preference = (targets[i] < targets[j]).to(scores.dtype)
    preference = torch.where(torch.isclose(targets[i], targets[j], atol=1e-8, rtol=0),
                             torch.full_like(preference, 0.5), preference)
    return F.binary_cross_entropy_with_logits(scores[j]-scores[i], preference)


def tensors(features):
    return tuple(torch.as_tensor(x, dtype=torch.float32) for x in features)
