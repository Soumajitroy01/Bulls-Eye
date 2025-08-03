import os
import torch
import networkx as nx
import matplotlib.pyplot as plt
from embedding_generator import visualize_embeddings
import numpy as np

# Paths
graph_name = "rca4"  # change this
graph_folder = "graph_data"
embedding_folder = "graph_embeddings"

nx_path = os.path.join(graph_folder, f"{graph_name}_nx.gpickle")
embedding_path = os.path.join(embedding_folder, f"{graph_name}_embeddings.npy")

# Load graph and embeddings
G = nx.read_gpickle(nx_path)
embeddings = np.load(embedding_path)

# Visualize
visualize_embeddings(G, embeddings, f"{graph_name}_viz.png")
print(f"Saved graph visualization to {graph_name}_viz.png")
