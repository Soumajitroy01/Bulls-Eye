import os
import torch
import pandas as pd
from torch_geometric.data import Dataset, Data, Batch

class RecipeTokenizer:
    def __init__(self, max_length=18):
        self.token_to_idx = {"<PAD>": 0, "<UNK>": 1}
        self.idx_to_token = {0: "<PAD>", 1: "<UNK>"}
        self.vocab_size = 2
        self.max_length = max_length
    
    def fit(self, recipes):
        unique_tokens = set()
        for recipe in recipes:
            if isinstance(recipe, str):
                tokens = recipe.split(';')
                for token in tokens:
                    token = token.strip()
                    if token:
                        unique_tokens.add(token)
        
        for token in unique_tokens:
            if token not in self.token_to_idx:
                self.token_to_idx[token] = self.vocab_size
                self.idx_to_token[self.vocab_size] = token
                self.vocab_size += 1
    
    def tokenize(self, recipe):
        tokens = recipe.split(';')
        token_ids = []
        
        for token in tokens:
            token = token.strip()
            if token:
                token_id = self.token_to_idx.get(token, self.token_to_idx["<UNK>"])
                token_ids.append(token_id)
        
        if len(token_ids) < self.max_length:
            token_ids += [self.token_to_idx["<PAD>"]] * (self.max_length - len(token_ids))
        else:
            token_ids = token_ids[:self.max_length]
        
        return torch.tensor(token_ids, dtype=torch.long)

class CircuitGraphDataset(Dataset):
    def __init__(self, csv_file, graph_dir, transform=None):
        self.data_df = pd.read_csv(csv_file)
        self.graph_dir = graph_dir
        self.transform = transform
        
        self.tokenizer = RecipeTokenizer()
        self.tokenizer.fit(self.data_df['recipe'].tolist())
        
    def __len__(self):
        return len(self.data_df)
    
    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()
        
        design_name = self.data_df.iloc[idx]['design']
        recipe = self.data_df.iloc[idx]['recipe']
        
        graph_file = design_name.replace('.bench', '.pt')
        graph_path = os.path.join(self.graph_dir, graph_file)
        
        try:
            graph_data = torch.load(graph_path)
        except FileNotFoundError:
            print(f"Warning: Graph file not found: {graph_path}")
            graph_data = Data(
                x=torch.zeros((1, 10)),
                edge_index=torch.zeros((2, 0), dtype=torch.long)
            )
        
        recipe_tokens = self.tokenizer.tokenize(recipe)
        
        target = {}
        for col in ['nodes', 'levels', 'iterations']:
            if col in self.data_df.columns:
                target[col] = torch.tensor([self.data_df.iloc[idx][col]], dtype=torch.float)
        
        sample = {
            'graph_data': graph_data,
            'recipe_tokens': recipe_tokens,
            'target': target
        }
        
        if self.transform:
            sample = self.transform(sample)
        
        return sample

def collate_fn(batch):
    graphs = [item['graph_data'] for item in batch]
    recipe_tokens = torch.stack([item['recipe_tokens'] for item in batch])
    
    targets = {}
    if 'target' in batch[0] and batch[0]['target']:
        for key in batch[0]['target'].keys():
            targets[key] = torch.cat([item['target'][key] for item in batch])
    
    batched_graphs = Batch.from_data_list(graphs)
    
    return {
        'graph_data': batched_graphs,
        'recipe_tokens': recipe_tokens,
        'target': targets
    }
