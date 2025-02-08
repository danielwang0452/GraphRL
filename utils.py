import torch_geometric as tg
import torch
import networkx as nx
import matplotlib.pyplot as plt

def rearrange_loaders(batches):
    '''
    obs: (T) array where each entry is a batch of N graphs
    We will return a single batch of N*T graphs
    '''
    all_data = []
    for batch in batches:
        # Convert the batch (which contains B graphs) to a list of Data objects
        data_list = batch.to_data_list()
        # Extend our overall list with these B graphs
        all_data.extend(data_list)
    combined_batch = tg.data.Batch.from_data_list(all_data)

    return combined_batch

def visualise_graph(edge_index):
    # Create an undirected graph using networkx
    G = nx.Graph()

    # The edge_index tensor is in COO format: each column [src, dst] is an edge.
    # Iterate over each column and add the edge to the graph.
    for src, dst in edge_index.t().tolist():
        G.add_edge(src, dst)

    # Visualize the graph
    plt.figure(figsize=(8, 6))
    nx.draw(G, with_labels=True, node_color="skyblue", node_size=500, edge_color="gray", font_weight='bold')
    plt.title("Visualized Graph from Edge Index")
    plt.show()
