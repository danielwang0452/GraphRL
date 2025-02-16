import torch_geometric as tg
import torch
import torch.nn as nn
import torch.nn.functional as F
from models import GAT
from env import sample_n_trajectories, eval_agent
from utils import rearrange_loaders, visualise_graph
from replay_buffer import Replay_Buffer
class DQNgent():
    def __init__(self, discount, N, T, n):
        self.critic = GAT(
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
        self.critic_optimizer = torch.optim.Adam(self.critic.parameters(), lr=1e-4)
        self.target_critic = GAT(
        in_channels=5,
        hidden_channels=64,
        num_layers=3,
        out_channels=3,
        embedding=True,
        readout=False
        )

    def get_action(self, batch): # observations is Batch object
        q_values = self.critic(
            batch.x,
            batch.edge_index,
            batch.batch
        ) # (B*n, d)
        actions = q_values.argmax(dim=1)
        # TODO: implement epsilon greedy exploration
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
        torch.save(self.critic.state_dict(), path)

    def load_state_dict(self, path):
        self.critic.load_state_dict(torch.load(path, weights_only=True))
        self.target_critic.load_state_dict(torch.load(path, weights_only=True))

    def update_critic(self, transitions):
        '''
        # transitions is a list of [(Data, Data_next), ... or (Data, None)]
        '''
        cur_obs = [transition[0] for transition in transitions]
        next_obs = [transition[1] for transition in transitions]
        # next_obs contains Data and None objects; we need to create a filler next_obs to pass
        # into the critic as a batch to compute next_q and a mask to mask out None
        # the filler will replace Nones with first graph in cur obs
        next_obs_filler = []
        none_indices = [] # also keep track of where Nones are located to create binary mask
        for g, graph in enumerate(next_obs):
            if graph == None:
                next_obs_filler.append(transitions[0][0])
                none_indices.append(g)
            else:
                next_obs_filler.append(graph)
        mask = torch.ones((len(transitions),))
        mask[none_indices] = 0
        # duplicate mask n times to mask each node
        mask = mask.unsqueeze(-1).repeat(1, self.n).flatten()

        cur_obs_batch = tg.data.Batch.from_data_list(cur_obs)
        next_obs_batch = tg.data.Batch.from_data_list(next_obs_filler)
        # get rewards
        rewards = torch.tensor([])
        for graph in cur_obs:
            rewards = torch.cat((rewards, graph.reward.repeat(n)))
        rewards = (rewards - rewards.mean())/rewards.std()
        # get actions
        # preprocessing done
        qa_values = self.critic(cur_obs_batch.x, cur_obs_batch.edge_index, cur_obs_batch.batch)
        next_qa_values = self.target_critic(next_obs_batch.x, next_obs_batch.edge_index, cur_obs_batch.batch)
        next_q_values = torch.max(next_qa_values, dim=1)[0] # shape (N*n)
        target_values = rewards + self.discount*mask*next_q_values
        q_values = qa_values.gather(dim=1, index=cur_obs_batch.action.unsqueeze(dim=1))
        loss = F.mse_loss(q_values, target_values)
        self.critic_optimizer.zero_grad()
        loss.backward()
        self.critic_optimizer.step()
        return loss.item()

    def update_target_critic(self):
        self.target_critic.load_state_dict(self.critic.state_dict())

N=128
T=7
n=30
discount=0.99
path = '/Users/danielwang/PycharmProjects/GraphRL/state_dicts/DQN.pth'
agent = DQNgent(discount, N, T, n)
replay_buffer = Replay_Buffer(capacity=50000)
replay_buffer.load(path='replay_buffer.pth')
agent.load_state_dict(path)
for i in range(1000000):
    if i%100 == 0:
        print(i)
        rewards = eval_agent(agent, N, T, n)
        print(rewards[:, -1].mean())
        agent.save_state_dict(path)
        agent.update_target_critic()
        replay_buffer.save_graphs(path='replay_buffer.pth')
    batches = sample_n_trajectories(agent, N, T, n, visualise=False)
    replay_buffer.insert_batches(batches)
    sampled_transitions = replay_buffer.sample_n_graphs(100)
    # a list of [(Data, Data_next), or (Data, None)]
    #print(sampled_transitions)
    agent.update_critic(sampled_transitions)
batches = sample_n_trajectories(agent, 1, T, n, visualise=True)

