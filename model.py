import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GCNConv, global_max_pool, global_mean_pool

class AIGEmbeddingNetwork(nn.Module):
    def __init__(self, node_feature_dim, hidden_dim=64):
        super(AIGEmbeddingNetwork, self).__init__()
        self.node_embedding = nn.Linear(node_feature_dim, hidden_dim)
        
        self.gcn1 = GCNConv(hidden_dim, hidden_dim)
        self.bn1 = nn.BatchNorm1d(hidden_dim)
        self.gcn2 = GCNConv(hidden_dim, hidden_dim)
        self.bn2 = nn.BatchNorm1d(hidden_dim)
        
        self.final_projection = nn.Linear(hidden_dim * 2, 128)
    
    def forward(self, x, edge_index, batch):
        x = self.node_embedding(x)
        
        x = self.gcn1(x, edge_index)
        x = self.bn1(x)
        x = F.relu(x)
        
        x = self.gcn2(x, edge_index)
        x = self.bn2(x)
        x = F.relu(x)
        
        max_pooled = global_max_pool(x, batch)
        mean_pooled = global_mean_pool(x, batch)
        
        pooled = torch.cat([max_pooled, mean_pooled], dim=1)
        
        aig_embedding = self.final_projection(pooled)
        
        return aig_embedding

class RecipeEmbeddingNetwork(nn.Module):
    def __init__(self, vocab_size, embedding_dim=60):
        super(RecipeEmbeddingNetwork, self).__init__()
        self.vocab_size = vocab_size
        
        # FC layer that transforms the entire recipe (1×18) to a single embedding (1×60)
        self.fc_embedding = nn.Linear(18, embedding_dim)  # Takes entire sequence as input
        
        # Convolutional layers with parameters from the image
        # For Conv1d, the output length is calculated as: 
        # out_length = (in_length - kernel_size) / stride + 1
        
        # For input length of 60:
        # K=12, S=3: out_length = (60-12)/3 + 1 = 17
        # K=15, S=3: out_length = (60-15)/3 + 1 = 16
        # K=18, S=3: out_length = (60-18)/3 + 1 = 15
        # K=21, S=4: out_length = (60-21)/4 + 1 = 10.75 ≈ 11 (not 14 as in the image)
        
        self.conv1 = nn.Conv1d(1, 1, kernel_size=12, stride=3)  # Output length will be 17
        self.conv2 = nn.Conv1d(1, 1, kernel_size=15, stride=3)  # Output length will be 16
        self.conv3 = nn.Conv1d(1, 1, kernel_size=18, stride=3)  # Output length will be 15
        
        # For K=21, S=4 to get output length 14, we need to adjust:
        # To get out_length = 14: in_length = (out_length-1)*stride + kernel_size
        # in_length = (14-1)*4 + 21 = 73
        # So we need padding of (73-60) = 13
        self.conv4 = nn.Conv1d(1, 1, kernel_size=21, stride=4, padding=13)  # Output length will be 14

    def forward(self, x):
        # x shape: [batch_size, seq_len] (where seq_len should be 18)
        batch_size = x.shape[0]
        # print(x)
        
        # Apply FC layer to get recipe embedding of shape [batch_size, embedding_dim]
        x_embedded = self.fc_embedding(x.float())  # [batch_size, embedding_dim]
        
        # Reshape for Conv1d operations - each embedding becomes a sequence of length 60 with 1 channel
        x_for_conv = x_embedded.unsqueeze(1)  # [batch_size, 1, embedding_dim]

        # Apply different convolutional filters
        x1 = F.relu(self.conv1(x_for_conv))  # [batch_size, 1, 17]
        x2 = F.relu(self.conv2(x_for_conv))  # [batch_size, 1, 16]
        x3 = F.relu(self.conv3(x_for_conv))  # [batch_size, 1, 15]
        x4 = F.relu(self.conv4(x_for_conv))  # [batch_size, 1, 14]
        
        # Flatten outputs (preserving batch dimension)
        x1_flat = x1.reshape(batch_size, -1)  # [batch_size, 17]
        x2_flat = x2.reshape(batch_size, -1)  # [batch_size, 16]
        x3_flat = x3.reshape(batch_size, -1)  # [batch_size, 15]
        x4_flat = x4.reshape(batch_size, -1)  # [batch_size, 14]
        
        # Concatenate all outputs
        x_concat = torch.cat([x1_flat, x2_flat, x3_flat, x4_flat], dim=1)  # [batch_size, 17+16+15+14=62]

        return x_concat


class QoRPredictionModel(nn.Module):
    def __init__(self, node_features, vocab_size, hidden_dim=512, output_dim=1):
        super(QoRPredictionModel, self).__init__()
        
        self.aig_embedding_net = AIGEmbeddingNetwork(node_features, hidden_dim=64)
        self.recipe_embedding_net = RecipeEmbeddingNetwork(vocab_size, embedding_dim=60)
        
        self.fc1 = nn.Linear(hidden_dim, 512)
        self.fc2 = nn.Linear(512, 512)
        self.fc3 = nn.Linear(512, 512)
        self.fc_out = nn.Linear(512, output_dim)
        
    def forward(self, graph_data, recipe_tokens):
        aig_embedding = self.aig_embedding_net(graph_data.x, graph_data.edge_index, graph_data.batch)
        recipe_features = self.recipe_embedding_net(recipe_tokens)
        
        combined_embedding = torch.cat([aig_embedding, recipe_features], dim=1)
        
        x = F.relu(self.fc1(combined_embedding))
        x = F.relu(self.fc2(x))
        x = F.relu(self.fc3(x))
        qor_prediction = self.fc_out(x)
        
        return qor_prediction
