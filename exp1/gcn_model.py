import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GCNConv

class AIGGCN(nn.Module):
    """Graph Convolutional Network for AIG embeddings"""
    
    def __init__(self, in_channels, hidden_channels, out_channels, num_layers=3):
        super(AIGGCN, self).__init__()
        
        self.num_layers = num_layers
        self.convs = nn.ModuleList()
        
        # First layer
        self.convs.append(GCNConv(in_channels, hidden_channels))
        
        # Hidden layers
        for _ in range(num_layers - 2):
            self.convs.append(GCNConv(hidden_channels, hidden_channels))
        
        # Output layer
        self.convs.append(GCNConv(hidden_channels, out_channels))
        
        # Edge attention for handling inversions
        self.edge_mlp = nn.Sequential(
            nn.Linear(1, 8),
            nn.ReLU(),
            nn.Linear(8, 1),
            nn.Sigmoid()
        )
    
    def forward(self, x, edge_index, edge_attr=None):
        # Process edge attributes if provided
        edge_weight = None
        if edge_attr is not None:
            edge_weight = self.edge_mlp(edge_attr).squeeze(-1)
        
        # Apply GCN layers
        for i in range(self.num_layers - 1):
            x = self.convs[i](x, edge_index, edge_weight)
            x = F.relu(x)
            x = F.dropout(x, p=0.2, training=self.training)
        
        # Final layer
        x = self.convs[-1](x, edge_index, edge_weight)
        
        return x