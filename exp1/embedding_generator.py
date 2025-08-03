import os
import torch
import numpy as np
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
import networkx as nx
from gcn_model import AIGGCN

# Check if CUDA is available
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {device}")

def generate_embeddings(data, embedding_dim=128, hidden_dim=128, num_layers=3):
    # Move data to device
    data = data.to(device)
    
    # Initialize GCN model
    model = AIGGCN(
        in_channels=data.x.size(1),  # Input features size
        hidden_channels=hidden_dim,
        out_channels=embedding_dim,
        num_layers=num_layers
    ).to(device)  # Move model to device
    
    # Generate node embeddings
    model.eval()
    with torch.no_grad():
        node_embeddings = model(data.x, data.edge_index)
        
    # Apply mean pooling to get a fixed-size vector
    mean_pooled_embedding = torch.mean(node_embeddings, dim=0)
    
    # Use a projection layer to ensure the embedding size is 128
    projection_layer = torch.nn.Linear(mean_pooled_embedding.shape[0], embedding_dim).to(device)
    projected_embedding = projection_layer(mean_pooled_embedding)
    
    # Convert embeddings to numpy for further processing
    embeddings_np = projected_embedding.unsqueeze(0).detach().cpu().numpy()  # Add batch dimension
    
    return embeddings_np


def generate_embeddings_from_folder(graph_data_folder, output_folder, embedding_dim=128):
    """Generate embeddings for all graphs in a folder"""
    
    if not os.path.exists(graph_data_folder):
        raise FileNotFoundError(f"Graph data folder {graph_data_folder} does not exist")
    
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)
    
    # Process each PyG data file
    for filename in os.listdir(graph_data_folder):
        if filename.endswith("_pyg.pt"):
            circuit_name = filename.replace("_pyg.pt", "")
            graph_path = os.path.join(graph_data_folder, filename)
            # info_path = os.path.join(graph_data_folder, f"{circuit_name}_info.json")
            # nx_path = os.path.join(graph_data_folder, f"{circuit_name}_nx.gpickle")
            
            print(f"Generating embeddings for {circuit_name}...")
            
            # Load PyG data
            data = torch.load(graph_path)
            
            # Generate embeddings
            embeddings = generate_embeddings(data, embedding_dim=embedding_dim)

            # Print the shape of the embeddings
            print(f"Embedding shape: {embeddings.shape}")
            
            # Save embeddings
            embedding_path = os.path.join(output_folder, f"{circuit_name}_embeddings.npy")
            np.save(embedding_path, embeddings)
            
            # Visualize if NetworkX graph is available
            # if os.path.exists(nx_path):
            #     G = nx.read_gpickle(nx_path)
            #     visualize_embeddings(G, embeddings, os.path.join(output_folder, f"{circuit_name}_viz.png"))
            
            print(f"  - Saved embeddings for {circuit_name}, shape: {embeddings.shape}")
    
    print("Embedding generation complete!")

def visualize_embeddings(G, embeddings, output_path):
    try:
        # Use spring layout to position nodes
        pos = nx.spring_layout(G)
        
        # Plot
        plt.figure(figsize=(12, 10))
        
        # Create node colors based on type
        node_colors = []
        for node in G.nodes():
            if G.nodes[node].get('is_input', False):
                node_colors.append('blue')
            elif G.nodes[node].get('is_output', False):
                node_colors.append('red')
            else:
                node_colors.append('green')
        
        # Create edge colors based on inversion
        edge_colors = ['red' if G.edges[edge].get('inverted', False) else 'black' for edge in G.edges()]
        
        nx.draw(G, pos, with_labels=True, node_color=node_colors, edge_color=edge_colors,
                labels={node: G.nodes[node]['name'] for node in G.nodes()})
        
        # Add legend
        from matplotlib.lines import Line2D
        legend_elements = [
            Line2D([0], [0], marker='o', color='w', markerfacecolor='blue', markersize=10, label='Input'),
            Line2D([0], [0], marker='o', color='w', markerfacecolor='red', markersize=10, label='Output'),
            Line2D([0], [0], marker='o', color='w', markerfacecolor='green', markersize=10, label='Internal Node'),
            Line2D([0], [0], color='black', lw=2, label='Regular Connection'),
            Line2D([0], [0], color='red', lw=2, label='Inverted Connection')
        ]
        plt.legend(handles=legend_elements, loc='upper right')
        
        plt.title("AND-INV Graph Visualization")
        plt.savefig(output_path)
        plt.close()
        
    except Exception as e:
        print(f"Visualization error: {e}")



def main():
    # Example usage
    graph_data_folder = "graph_data"
    embedding_output_folder = "embeddings"
    
    if not os.path.exists(graph_data_folder):
        print(f"Error: Graph data folder '{graph_data_folder}' does not exist.")
        print("Please run graph_generator.py first to create graph data.")
        return
    
    generate_embeddings_from_folder(graph_data_folder, embedding_output_folder)

if __name__ == "__main__":
    main()