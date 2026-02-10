import pandas as pd
import networkx as nx
import torch
from torch_geometric.data import Data

def build_graph(csv_path):
    df = pd.read_csv(csv_path)

    G = nx.DiGraph()
    for _, row in df.iterrows():
        G.add_edge(row["src"], row["dst"], amount=row["amount"])

    nodes = list(G.nodes)
    node_index = {node: i for i, node in enumerate(nodes)}

    edge_index = []
    edge_attr = []

    for u, v, attr in G.edges(data=True):
        edge_index.append([node_index[u], node_index[v]])
        edge_attr.append([attr["amount"]])

    edge_index = torch.tensor(edge_index, dtype=torch.long).t().contiguous()
    edge_attr = torch.tensor(edge_attr, dtype=torch.float)

    x = torch.ones((len(nodes), 1))  # node features (dummy for demo)

    labels = []
    for node in nodes:
        fraud = df[(df.src == node) | (df.dst == node)]["label"].max()
        labels.append(fraud)

    y = torch.tensor(labels, dtype=torch.long)

    return Data(x=x, edge_index=edge_index, edge_attr=edge_attr, y=y)