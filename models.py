import torch_geometric as tg
import torch
import torch.nn as nn
import torch.nn.functional as F

class GAT(nn.Module):
    def __init__(self, in_channels, hidden_channels, num_layers, out_channels,
                 embedding=True, readout=False, dropout=0.5):
        super().__init__()
        self.embedding, self.readout = embedding, readout
        if readout:
            gat_out_channels = hidden_channels
            self.out_net = nn.Linear(hidden_channels, out_channels)
        else:
            gat_out_channels = out_channels
        if embedding:
            self.embedding = nn.Embedding(in_channels, hidden_channels)
            gat_in_channels = hidden_channels
        else:
            gat_in_channels = in_channels
        self.gat = tg.nn.GAT(
        in_channels=gat_in_channels,
        hidden_channels=hidden_channels,     # or any desired hidden size
        num_layers=num_layers,           # number of GATConv layers
        out_channels=gat_out_channels,
        dropout=0.0,
        act='relu',
        jk=None  # if using jump knowledge, else leave it as default
        )

    def forward(self, x, edge_index, batch):
        if self.embedding:
            x = self.embedding(x)
        features = self.gat(x, edge_index, batch)
        if self.readout:
            pooled = tg.nn.global_mean_pool(features, batch)
            out = self.out_net(pooled)
            return out
        return features