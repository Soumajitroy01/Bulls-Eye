import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader

# Define vocabulary
vocabulary = set()
for k in range(4, 17, 2):  # K values from 4 to 16
    for n in range(1, 4):  # N values from 1 to 3
        vocabulary.add(f"rs -K {k} -N {n}")
        vocabulary.add(f"rs -K {k}")  # Add without N
        vocabulary.add(f"rs -N {n}")  # Add without K
vocabulary.add("rs")  # Add rs without any parameters

# Add tokens without parameters
vocabulary.update(["b", "rw", "rwz", "rf", "rfz"])

# Create a dictionary mapping tokens to indices
token_to_idx = {token: i for i, token in enumerate(vocabulary)}
idx_to_token = {i: token for token, i in token_to_idx.items()}

# Sample recipe string
recipe_string = "b; rs -K 6; rw; rs -K 6 -N 2; rf; rs -K 8; b; rs -K 8 -N 2; rw; rs -K 10; rwz; rs -K 10 -N 2; b; rs -K 12; rfz; rs -K 12 -N 2; rwz; b"

# Split recipe into tokens
recipe_tokens = recipe_string.split(';')

# Preprocess tokens to match vocabulary format
preprocessed_tokens = []
for token in recipe_tokens:
    token = token.strip()
    if token.startswith("rs"):
        parts = token.split()
        base_token = parts[0]
        
        # Check for -K and -N parameters
        k_value = None
        n_value = None
        for part in parts[1:]:
            if part.startswith("-K"):
                k_value = part.split("-K")[1]
            elif part.startswith("-N"):
                n_value = part.split("-N")[1]
        
        # Construct the preprocessed token
        if k_value and n_value:
            preprocessed_token = f"{base_token} -K {k_value} -N {n_value}"
        elif k_value:
            preprocessed_token = f"{base_token} -K {k_value}"
        elif n_value:
            preprocessed_token = f"{base_token} -N {n_value}"
        else:
            preprocessed_token = base_token
        
        preprocessed_tokens.append(preprocessed_token)
    else:
        preprocessed_tokens.append(token)

# Convert tokens to indices
token_indices = [token_to_idx[token] for token in preprocessed_tokens]

# Define a custom dataset class for recipes
class RecipeDataset(Dataset):
    def __init__(self, token_indices):
        self.token_indices = token_indices
    
    def __len__(self):
        return len(self.token_indices)
    
    def __getitem__(self, idx):
        return torch.tensor(self.token_indices[idx])

# Create a custom BERT-like model
class CustomBertModel(nn.Module):
    def __init__(self, vocab_size, embedding_dim, num_heads, num_layers):
        super(CustomBertModel, self).__init__()
        self.embedding = nn.Embedding(vocab_size, embedding_dim)
        encoder_layer = nn.TransformerEncoderLayer(d_model=embedding_dim, nhead=num_heads, dim_feedforward=embedding_dim, dropout=0.1)
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        self.fc = nn.Linear(embedding_dim, embedding_dim)
    
    def forward(self, token_indices):
        # Embed tokens
        embeddings = self.embedding(token_indices)
        
        # Apply transformer encoder
        encoder_output = self.encoder(embeddings.unsqueeze(1)).squeeze(1)
        
        # Average embeddings across all tokens
        averaged_embedding = torch.mean(encoder_output, dim=0)
        
        # Apply a fully connected layer to refine the embedding
        refined_embedding = torch.relu(self.fc(averaged_embedding))
        
        return refined_embedding

# Initialize the model
model = CustomBertModel(len(vocabulary), embedding_dim=64, num_heads=8, num_layers=3)

# Create a dataset and data loader
dataset = RecipeDataset(token_indices)
dataloader = DataLoader(dataset, batch_size=1, shuffle=False)

# Train the model (optional)
# optimizer = optim.Adam(model.parameters(), lr=0.001)
# criterion = nn.CrossEntropyLoss()

# for epoch in range(10):  # Example training loop
#     for batch in dataloader:
#         optimizer.zero_grad()
#         outputs = model(batch)
#         loss = criterion(outputs, torch.zeros_like(outputs))  # Dummy loss for demonstration
#         loss.backward()
#         optimizer.step()
#     print(f"Epoch {epoch+1}, Loss: {loss.item()}")

# Generate embeddings
with torch.no_grad():
    embeddings = []
    for batch in dataloader:
        embedding = model(batch)
        embeddings.append(embedding)
    
    # Average embeddings across all tokens (if needed)
    averaged_embedding = torch.mean(torch.stack(embeddings), dim=0)

    # Reshape to [1, 64]
    averaged_embedding = averaged_embedding.unsqueeze(0)
    
    print(averaged_embedding.shape)

# Save embeddings
torch.save(averaged_embedding, "recipe_embedding.pt")

# To load embeddings later
loaded_embedding = torch.load("recipe_embedding.pt")
print(loaded_embedding.shape)
