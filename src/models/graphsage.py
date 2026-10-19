"""
Two-layer GraphSAGE network for node-level structural-deprivation
classification (proposal Sec 3.2.3 / FR-04).

Mean aggregation is the proposal's stated choice: it is the inductive
Sample-and-Aggregate variant that generalises to unseen buildings without
retraining static embeddings -- important in settlements whose footprints
change between OSM extracts. Hidden width defaults to 64; the Sprint 3
notebook can sweep this without touching the architecture class.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import SAGEConv

from .. import config
from ..utils.deprivation_label import NODE_FEATURE_NAMES


class GraphSAGE(nn.Module):
    """
    Two SAGEConv layers with ReLU + dropout between them, ending in a
    linear projection to ``num_classes`` logits. Matches the proposal's
    "two-layer GraphSAGE network with mean aggregation."
    """

    def __init__(
        self,
        in_channels: int = None,
        hidden_channels: int = 64,
        num_classes: int = None,
        dropout: float = 0.5,
        aggr: str = "mean",
    ):
        super().__init__()
        if in_channels is None:
            in_channels = len(NODE_FEATURE_NAMES)
        if num_classes is None:
            num_classes = config.DEPRIVATION_NUM_CLASSES

        self.conv1 = SAGEConv(in_channels, hidden_channels, aggr=aggr)
        self.conv2 = SAGEConv(hidden_channels, hidden_channels, aggr=aggr)
        self.classifier = nn.Linear(hidden_channels, num_classes)
        self.dropout = dropout

    def forward(self, x: torch.Tensor, edge_index: torch.Tensor) -> torch.Tensor:
        x = self.conv1(x, edge_index)
        x = F.relu(x)
        x = F.dropout(x, p=self.dropout, training=self.training)
        x = self.conv2(x, edge_index)
        x = F.relu(x)
        x = F.dropout(x, p=self.dropout, training=self.training)
        return self.classifier(x)

    def encode(self, x: torch.Tensor, edge_index: torch.Tensor) -> torch.Tensor:
        """
        Return the penultimate node embeddings (pre-classifier). Useful for
        SHAP / neighbourhood diagnostics in Sprint 6 without re-running the
        full forward pass structure.
        """
        x = self.conv1(x, edge_index)
        x = F.relu(x)
        x = self.conv2(x, edge_index)
        return F.relu(x)
