"""Graph neural network for docking-score regression.

Design note
-----------
Architecture: a GINE encoder (GIN with edge features) over RDKit molecular
graphs, followed by concatenated mean and max pooling and an MLP head.

Why GINE rather than GCN or GAT. Docking scores depend strongly on substructure
*counts*: how many hydrogen-bond donors reach the pocket, how many rotatable
bonds pay an entropic penalty, how much hydrophobic surface is buried. GIN is
the message-passing scheme with provably maximal discriminative power for
exactly that kind of multiset-counting question, and GINE is the variant that
consumes edge features, which here encode bond order and ring membership. GCN's
degree normalisation actively discards the count information that matters.

Why mean *and* max pooling. Mean pooling tracks average character, which
correlates with size-driven terms; max pooling preserves "does this molecule
contain at least one of X", which is what a single strong interaction looks
like. Concatenating both is cheap and consistently beat either alone in
development.

Dropout placement is deliberate and is the thing MC-dropout depends on. It is
applied after every message-passing block, not only in the head. Dropout in
the head alone would make MC-dropout sample only readout noise while the
learned representation stayed fixed, which produces confidently wrong
uncertainty on exactly the out-of-distribution molecules where uncertainty
matters most.

The model outputs a single scalar in *scaled* target space; the inverse
transform lives with the dataset's scaler, not in the model.
"""

from __future__ import annotations

from typing import Any

import torch
from torch import Tensor, nn
from torch_geometric.nn import GINEConv, global_add_pool, global_max_pool, global_mean_pool

from ..config import ModelConfig
from ..data.featurise import ATOM_FEATURE_DIM, BOND_FEATURE_DIM, FEATURISER_VERSION


def _mlp(in_dim: int, hidden_dim: int, out_dim: int) -> nn.Sequential:
    """Two-layer MLP used inside each GINE block."""
    return nn.Sequential(
        nn.Linear(in_dim, hidden_dim),
        nn.BatchNorm1d(hidden_dim),
        nn.ReLU(),
        nn.Linear(hidden_dim, out_dim),
    )


class DockingGNN(nn.Module):
    """GINE encoder with mean+max pooling and an MLP regression head."""

    def __init__(
        self,
        cfg: ModelConfig,
        atom_dim: int = ATOM_FEATURE_DIM,
        bond_dim: int = BOND_FEATURE_DIM,
    ) -> None:
        super().__init__()
        self.cfg = cfg
        self.atom_dim = atom_dim
        self.bond_dim = bond_dim
        hidden = cfg.hidden_dim

        self.atom_encoder = nn.Linear(atom_dim, hidden)
        # GINEConv requires edge features to match the node feature width.
        self.bond_encoder = nn.Linear(bond_dim, hidden)

        self.convs = nn.ModuleList()
        self.norms = nn.ModuleList()
        for _ in range(cfg.num_layers):
            self.convs.append(GINEConv(_mlp(hidden, hidden, hidden), train_eps=True))
            self.norms.append(nn.BatchNorm1d(hidden))

        self.dropout = nn.Dropout(cfg.dropout)

        pooled_dim = hidden * (2 if cfg.pool == "mean_max" else 1)
        self.head = nn.Sequential(
            nn.Linear(pooled_dim, hidden),
            nn.ReLU(),
            nn.Dropout(cfg.dropout),
            nn.Linear(hidden, hidden // 2),
            nn.ReLU(),
            nn.Dropout(cfg.dropout),
            nn.Linear(hidden // 2, 1),
        )

    def forward(self, x: Tensor, edge_index: Tensor, edge_attr: Tensor, batch: Tensor) -> Tensor:
        """Predict a scaled docking score per graph.

        Args:
            x: Node features ``[n_nodes, atom_dim]``.
            edge_index: COO edges ``[2, n_edges]``.
            edge_attr: Edge features ``[n_edges, bond_dim]``.
            batch: Graph assignment per node ``[n_nodes]``.

        Returns:
            Tensor of shape ``[n_graphs]``.
        """
        h = self.atom_encoder(x)
        e = self.bond_encoder(edge_attr)

        for conv, norm in zip(self.convs, self.norms, strict=True):
            h_in = h
            h = conv(h, edge_index, e)
            h = norm(h)
            h = torch.relu(h)
            # Dropout inside the encoder: this is what MC-dropout samples.
            h = self.dropout(h)
            if self.cfg.residual:
                h = h + h_in

        pooled = self._pool(h, batch)
        return self.head(pooled).squeeze(-1)

    def _pool(self, h: Tensor, batch: Tensor) -> Tensor:
        if self.cfg.pool == "mean_max":
            return torch.cat([global_mean_pool(h, batch), global_max_pool(h, batch)], dim=-1)
        if self.cfg.pool == "mean":
            return global_mean_pool(h, batch)
        if self.cfg.pool == "max":
            return global_max_pool(h, batch)
        if self.cfg.pool == "sum":
            return global_add_pool(h, batch)
        raise ValueError(f"unknown model.pool {self.cfg.pool!r}")

    def enable_mc_dropout(self) -> None:
        """Put dropout layers in training mode while the rest stays in eval.

        This is the whole trick behind MC-dropout at inference: BatchNorm must
        keep using its running statistics (otherwise predictions would depend
        on how the inference batch happened to be composed), while dropout must
        stay stochastic. Calling ``model.train()`` would wrongly do both.
        """
        self.eval()
        for module in self.modules():
            if isinstance(module, nn.Dropout):
                module.train()

    def architecture_summary(self) -> dict[str, Any]:
        """Return a checkpoint/manifest description of this model."""
        return {
            "class": type(self).__name__,
            "hidden_dim": self.cfg.hidden_dim,
            "num_layers": self.cfg.num_layers,
            "dropout": self.cfg.dropout,
            "pool": self.cfg.pool,
            "residual": self.cfg.residual,
            "atom_feature_dim": self.atom_dim,
            "bond_feature_dim": self.bond_dim,
            "featuriser_version": FEATURISER_VERSION,
            "n_parameters": sum(p.numel() for p in self.parameters()),
        }


def build_model(cfg: ModelConfig) -> DockingGNN:
    """Construct a model with the featuriser's current dimensions."""
    return DockingGNN(cfg)
