import torch_geometric as tg
import torch
import torch.nn as nn
import torch.nn.functional as F
from models import GAT
from env import sample_n_trajectories, eval_agent
from utils import rearrange_loaders, visualise_graph

class PGAgent():
    def __init__(self, discount, N, T, n):
        self.actor = GAT(
        in_channels=5,
        hidden_channels=64,
        num_layers=3,
        out_channels=3,
        embedding=True,
        readout=False
        )
        self.discount = discount
        self.N = N
        self.T = T
        self.n = n
        self.optimizer = torch.optim.Adam(self.actor.parameters(), lr=1e-4)

    def get_action(self, batch): # observations is Batch object
        action_logits = self.actor(
            batch.x,
            batch.edge_index,
            batch.batch
        ) # (B*n, d)
        action_probs = F.softmax(action_logits, dim=1)
        action_dist = torch.distributions.Categorical(probs=action_probs)
        actions = action_dist.sample()
        return actions

    def compute_advantages(self, batch):
        '''
        :param rewards: shape (N*T)
        :return: shape (N, T)
        '''
        # reshape rewards to (N, T)
        rewards = batch.reward.reshape(self.T, self.N).permute(1, 0)
        # discounted reward to go estimator
        discounted_rewards_to_go = torch.zeros_like(rewards)
        discounted_reward_to_go = torch.zeros((self.N,))
        for i in range(self.T - 1, -1, -1):
            discounted_reward_to_go = discounted_reward_to_go * self.discount + rewards[:, i]
            discounted_rewards_to_go[:, i] = discounted_reward_to_go
        #discounted_rewards_to_go.reverse()
        # advantage normalisation
        #print(rewards, rewards.sum(dim=1).unsqueeze(-1).repeat((1, self.T)))
        mean = discounted_rewards_to_go.mean()#.unsqueeze(dim=1).repeat((1, self.T))
        std = discounted_rewards_to_go.std()#.unsqueeze(dim=1).repeat((1, self.T))
        discounted_rewards_to_go = (discounted_rewards_to_go - mean)/(std+1e-8)
        #a = rewards.sum(dim=1).unsqueeze(-1).repeat((1, self.T))

        #return a
        return discounted_rewards_to_go

    def save_state_dict(self, path):
        torch.save(self.actor.state_dict(), path)

    def load_state_dict(self, path):
        self.actor.load_state_dict(torch.load(path, weights_only=True))

    def update_actor(self, batches):
        '''
        batch contains attributes
        x: colour indices (N*n*T)
        action: action index (N*n*T)
        reward: (N*T)
        '''
        # print(edge_index)
        batch = rearrange_loaders(batches) # make single loader with a batch of T*N
        # Compute target values
        #print(batch.x, batch.action, batch.reward)
        #print(batch.edge_index)
        advantages = self.compute_advantages(batch) # (N, T)
        #q_values = [np.array(self._discounted_reward_to_go(reward)) for reward in rewards]

        logits = self.actor(batch.x, batch.edge_index, batch.batch)
        # get per-node advantage values - copy per-graph value to each node
        #print(advantages)
        advantages =  1*advantages.unsqueeze(-1).repeat(1, 1, self.n).permute(1, 0, 2).flatten()
        # logits and actions are arranged (b_0 t_0, b_1 t_0, b_0 t_1, ... b_N, t_T) in dim 0
        #visualise_graph(batch.edge_index)
        #print(batch.x, batch.reward, advantages)
        #advantages = batch.x
        loss = (F.cross_entropy(logits, batch.action, reduction='none')*advantages).mean()
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        return loss.item()

N=128
T=7
n=7
discount=0.99
path = '/Users/danielwang/PycharmProjects/GraphRL/state_dicts/actor.pth'
agent = PGAgent(discount, N, T, n)
agent.load_state_dict(path)
for i in range(1):
    if i%100 == 0:
        print(i)
        rewards = eval_agent(agent, N, T, n)
        print(rewards[:, -1].mean())
        agent.save_state_dict(path)
    batches = sample_n_trajectories(agent, N, T, n, visualise=False)
    agent.update_actor(batches)
batches = sample_n_trajectories(agent, 1, T, n, visualise=True)
