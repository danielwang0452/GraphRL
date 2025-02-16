import torch_geometric as tg
import torch
import torch.nn as nn
import torch.nn.functional as F
from models import GAT
from env import sample_n_trajectories, eval_agent
from utils import rearrange_loaders, visualise_graph
import numpy as np


class Replay_Buffer():
    def __init__(self, capacity): # total number of graphs
        self.capacity = capacity
        self.current_idx = 0
        self.graphs = [None for i in range(capacity)]

    def insert_batches(self, batches):
        '''
        :param batches: a trajectory list [DataBatch, ..., DataBatch] of length T
        we will split it into N*T individual graphs and insert it into the replay buffer.
        store as (graph, next graph) pairs
        '''
        next_batches = batches[1:]
        for t, batch in enumerate(batches): # iterates over time step
            datas = batch.to_data_list()
            if t < len(batches)-1: # last timestep is None
                next_datas = next_batches[t].to_data_list()
            elif  t == len(batches)-1:
                next_datas = [None for _ in range(len(datas))]
            for b, data in enumerate(datas): # iterates over batch size
                self.graphs[self.current_idx % self.capacity] = (data, next_datas[b])
                self.current_idx += 1
        #print(self.graphs, len(self.graphs))

    def sample_n_graphs(self, n):
        indices = np.random.randint(0, min(self.current_idx, self.capacity), size=(n))
        #print(indices.tolist())
        #print([self.graphs[index] for index in indices])
        return [self.graphs[index] for index in indices]

    def save_graphs(self, path):
        torch.save(self.graphs, path)

    def load(self, path):
        self.graphs = torch.load(path)


