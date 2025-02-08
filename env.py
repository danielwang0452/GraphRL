import torch_geometric as tg
import torch
import networkx as nx
import matplotlib.pyplot as plt
import random
import numpy as np
import copy

color_palette = [0, 1, 2, 3, 4]
colors = ['#FF5733', '#33FF57', '#3357FF', '#F3FF33', '#FF33F6']
k = 7  # Number of edges
c = 5  # Number of distinct colors

def sample_n_trajectories(agent, N, T, n, visualise=False):
    '''
    :param N:
    :param T:
    :param agent:
    :return:
    obs: (T) array where each entry is a dataloader containing a batch of N graphs
    next_obs: same as obs
    acs: tensor shape (N*n, T)
    rewards: tensor (N, T)
    '''
    # initialise n graphs
    batches = []
    batch = sample_n_graphs(N, n) # 1 Batch.from_data_list() of N graphs
    for t in range(T):
        actions = agent.get_action(batch)
        new_batch = env_step(batch, actions)
        rewards = compute_rewards(batch)

        # assign actions, rewards to batch
        actions_2 = actions.reshape((N, n)) # N*n -> (N, n)
        data_list = batch.to_data_list()
        for d, data in enumerate(data_list):
            data.action = actions_2[d]
            data.reward = rewards[d]
        batch = tg.data.Batch.from_data_list(data_list)
        batches.append(batch)
        batch = new_batch
    if visualise: # draw a graph at timestep 0 and T
        data_0, data_T = batches[0].to_data_list()[0], batches[-1].to_data_list()[0]
        print(f'rewards: {data_0.reward}, {data_T.reward}')
        datas = [data_0, data_T]
        datas = []
        for b, batch in enumerate(batches):
            if b % ((len(batches))//5+1)==0:
                datas.append(batches[b].to_data_list()[0])
        print(len(datas))
        show_graphs(datas)
        '''
        for data in datas:
            G = tg.utils.to_networkx(data).to_undirected()
            node_colors = {node: colors[data.x[n]] for n, node in enumerate(G.nodes())}
            pos = nx.spring_layout(G)  # For consistent layout
            nx.draw(G, pos, with_labels=True, node_color=[node_colors[node] for node in G.nodes()],
                    edge_color='gray', node_size=500, font_color='white')

            # Add labels for component sizes
            node_colors_nums = {node: color_palette[data.x[n]] for n, node in enumerate(G.nodes())}
            labels = {node: f"{node_colors_nums[node]}" for node in G.nodes()}
            nx.draw_networkx_labels(G, pos, labels=labels, font_color='black', font_size=10, verticalalignment='bottom')

            plt.show()
        '''
    return batches #obs, next_obs, torch.cat(acs, dim=0),

def show_graphs(datas):
    num_graphs = len(datas)
    # Create a figure with one row and num_graphs columns
    fig, axes = plt.subplots(1, num_graphs, figsize=(5 * num_graphs, 5))

    # If there's only one graph, axes might not be a list:
    if num_graphs == 1:
        axes = [axes]

    for ax, data in zip(axes, datas):
        # Convert the data object to an undirected NetworkX graph
        G = tg.utils.to_networkx(data, node_attrs=['x']).to_undirected()

        # Create node color mapping: assuming data.x is a 1D tensor of color indices.
        node_colors = {node: colors[data.x[n]] for n, node in enumerate(G.nodes())}

        # Compute a layout for the graph
        pos = nx.spring_layout(G, seed=42)

        # Draw the graph on the given axis
        nx.draw(G, pos, with_labels=True, node_color=[node_colors[node] for node in G.nodes()],
                edge_color='gray', node_size=500, font_color='black', ax=ax)

        # If you want to add labels (for example, from color_palette)
        #node_labels = {node: f"{color_palette[data.x[n]]}" for n, node in enumerate(G.nodes())}
        #nx.draw_networkx_labels(G, pos, labels=None, font_color='black', font_size=10, ax=ax)

    # Display all subplots in one window
    plt.show()
    return

def env_step(batch, actions):
    '''
    :param observations: a Batch of size N
    :param actions: actions is shape (N, n) where each entry is [0, c)
    indicating how many places to rotate in the colour wheel
    action: cycle the colour indices given by actions
    env dynamics: after applying the action,
    a node with maximum degree will rewire an adge and sample a new colour accordin to probs
    :return: next_observations, rewards
    '''
    # apply action - there is only 1 batch
    new_batch = copy.deepcopy(batch)
    new_batch.x = (new_batch.x + actions) % c
    #
    datas = new_batch.to_data_list()
    for data in datas:
        source, target = data.edge_index[0, :].tolist(), data.edge_index[1, :].tolist()

        # find max node and keep track of its neighbours
        max_appearances = 0
        cur_appearances = 1
        current = source[0]
        neighbours = set()
        all_nodes = set(torch.arange(c).tolist())
        for n, num in enumerate(source):
            if n >= 1:
                if num == current:
                    cur_appearances += 1
                else:
                    cur_appearances = 1
                    current = num
                    neighbours = set()
            neighbours.add(target[n])
            if cur_appearances > max_appearances:
                max_index = n
                max_appearances = cur_appearances
        # set new colour at node x[max index]
        data.x[source[max_index]] = 0
        # rewire edge at max index
        neighbours.add(source[max_index])
        diff = all_nodes-neighbours
        if len(diff)>0:
            data.edge_index[1, max_index] = random.sample(diff, 1)[0]


    # TODO: after applying the action, a node with maximum degree will sample a new colour according to probs
    return new_batch

def compute_rewards(batch):
    # convert to networkx
    rewards = []
    for graph in batch.to_data_list():
        color_idx = graph.x
        G = tg.utils.to_networkx(graph).to_undirected()
        node_colors = {node: color_palette[color_idx[n]] for n, node in enumerate(G.nodes())}
        rewards.append(get_label(G, node_colors))
    return rewards # (N)

def sample_n_graphs(N, n): # returns dataloader
    data_list = []
    nums = []
    for i in range(N):
        G = nx.gnm_random_graph(n, k)#, seed=42)
        # Ensure the color palette has exactly c colors
        assert len(color_palette) == c, "The color palette must contain exactly c colors."

        # Assign a random color to each node
        color_idx = np.random.randint(0, c, size=(len(G.nodes())))
        node_colors = {node: color_palette[color_idx[n]] for n, node in enumerate(G.nodes())}

        # Initialize a dictionary to store the size of the connected component for each node
        component_sizes = {node: 0 for node in G.nodes()}
        data = tg.utils.from_networkx(G)
        data.x = torch.tensor(color_idx)
        data.action = torch.tensor(color_idx)
        data_list.append(data)
    batch = tg.data.Batch.from_data_list(data_list)
    return batch

def get_label(G, node_colors):
    component_sizes = {node: 0 for node in G.nodes()}
    for color in color_palette:
        # Create a subgraph containing only nodes of the current color
        color_subgraph = G.subgraph([node for node in G.nodes() if node_colors[node] == color])

        # Find connected components in the color subgraph
        for component in nx.connected_components(color_subgraph):
            component_size = len(component)
            for node in component:
                component_sizes[node] = component_size
    sum = 0.0
    for node in G.nodes():
        sum -= component_sizes[node]
    return sum#/len(G.nodes())

def eval_agent(agent, N, T, n):
    batches = sample_n_trajectories(agent, N, T, n)
    rewards = []
    for batch in batches:
        rewards.append(batch.reward)
    rewards = torch.stack(rewards, dim=1)
    return rewards