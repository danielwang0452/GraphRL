import torch_geometric as tg
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import networkx as nx
import matplotlib.pyplot as plt
import random
from sample_graph import sample_n_graphs

class GAT(nn.Module):
    def __init__(self, in_channels, hidden_channels, num_layers, out_channels, dropout=0.5):
        super().__init__()
        self.embedding = nn.Embedding(in_channels, in_channels)
        self.gat = tg.nn.GAT(
        in_channels=in_channels,
        hidden_channels=hidden_channels,     # or any desired hidden size
        num_layers=num_layers,           # number of GATConv layers
        out_channels=hidden_channels,
        dropout=0.5,
        act='relu',
        jk=None  # if using jump knowledge, else leave it as default
        )
        self.out_net = nn.Linear(hidden_channels, out_channels)
    def forward(self, x, edge_index, batch):
        embeddings = self.embedding(x)
        features = self.gat(embeddings, edge_index, batch)
        pooled = tg.nn.global_mean_pool(features, batch)
        out = self.out_net(pooled)
        return out

dtype = torch.float32
model = GAT(
    in_channels=5,
    hidden_channels=64,     # or any desired hidden size
    num_layers=3,           # number of GATConv layers
    out_channels=2
)

dataloader = sample_n_graphs(32)
model.train()
optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)
for epoch in range(10000):
    losses = []
    for batch in dataloader:
        #print(batch.x.shape)
        x = batch.x#.to(dtype)
        y = torch.tensor(batch.y, dtype=dtype)
        edge_index = torch.tensor(batch.edge_index)
        batch_info = torch.tensor(batch.batch)
        out = model(x, edge_index, batch=batch_info)
        mean = y.mean()
        labels = torch.zeros_like(y)
        labels[y>15.0] = torch.tensor(1.0)
        loss = F.cross_entropy(out, labels.to(torch.long))
        losses.append(loss.item())
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

   # print(np.array(losses).mean())

    #preds = F.softmax(out, dim=1)

