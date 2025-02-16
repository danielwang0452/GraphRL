import torch_geometric as tg
import torch
import networkx as nx
import matplotlib.pyplot as plt
import random
import numpy as np

color_palette = ['#FF5733', '#33FF57', '#3357FF', '#F3FF33', '#FF33F6']  # Example hex colors

def sample_n_graphs(N): # returns dataloader
    n = 10  # Number of nodes
    k = 40  # Number of edges
    c = 5  # Number of distinct colors
    data_list = []
    nums = []
    for i in range(N):
        G = nx.gnm_random_graph(n, k)#, seed=42)

        # Ensure the color palette has exactly c colors
        assert len(color_palette) == c, "The color palette must contain exactly c colors."

        # Assign a random color to each node

        color_idx = np.random.randint(4, 5, size=(n,))
        node_colors = {node: color_palette[color_idx[n]] for n, node in enumerate(G.nodes())}
        # Initialize a dictionary to store the size of the connected component for each node
        #component_sizes = {node: 0 for node in G.nodes()}
        label = get_label(G, node_colors)

        data = tg.utils.from_networkx(G)
        data.y = label
        nums.append(label)
        # one hot encode x based on colour labels
        #encoded_x = np.zeros((color_idx.size, c), dtype=int)
        #encoded_x[np.arange(color_idx.size), color_idx] = 1
        data.x = torch.tensor(color_idx)#encoded_x)
        data_list.append(data)
    #print(data_list)
    dataloader = tg.loader.DataLoader(data_list, batch_size=2, shuffle=True)
    #for batch in dataloader:
    #    print(batch)
    #    print(batch.ptr)
    return dataloader



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
    sum = 0
    for node in G.nodes():
        sum += component_sizes[node]
    return sum

def show_random_graph():
    # Parameters
    n = 20  # Number of nodes
    k = 40  # Number of edges
    c = 5  # Number of distinct colors

    # Generate random graph
    G = nx.gnm_random_graph(n, k, seed=42)

    # Define a list of c distinct colors
    color_palette = ['#FF5733', '#33FF57', '#3357FF', '#F3FF33', '#FF33F6']  # Example hex colors

    # Ensure the color palette has exactly c colors
    assert len(color_palette) == c, "The color palette must contain exactly c colors."

    # Assign a random color to each node
    node_colors = {node: random.choice(color_palette) for node in G.nodes()}

    # Initialize a dictionary to store the size of the connected component for each node
    component_sizes = {node: 0 for node in G.nodes()}

    # Process each color separately
    for color in color_palette:
        # Create a subgraph containing only nodes of the current color
        color_subgraph = G.subgraph([node for node in G.nodes() if node_colors[node] == color])

        # Find connected components in the color subgraph
        for component in nx.connected_components(color_subgraph):
            component_size = len(component)
            for node in component:
                component_sizes[node] = component_size

    # Draw the graph with node colors and labels indicating the component size
    pos = nx.spring_layout(G, seed=42)  # For consistent layout
    nx.draw(G, pos, with_labels=True, node_color=[node_colors[node] for node in G.nodes()],
            edge_color='gray', node_size=500, font_color='white')

    # Add labels for component sizes
    labels = {node: f"{component_sizes[node]}" for node in G.nodes()}
    nx.draw_networkx_labels(G, pos, labels=labels, font_color='black', font_size=10, verticalalignment='bottom')

    plt.show()

#sample_n_graphs(10)