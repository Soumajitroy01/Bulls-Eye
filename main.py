import os
import argparse
import torch
from dataset import CircuitGraphDataset
from model import QoRPredictionModel
from train import train_and_evaluate, evaluate_only

def parse_args():
    parser = argparse.ArgumentParser(description='Train QoR prediction model')
    parser.add_argument('--csv_file', type=str, required=True, help='Path to CSV file with recipes')
    parser.add_argument('--graph_dir', type=str, required=True, help='Directory containing graph .pt files')
    parser.add_argument('--output_dir', type=str, default='./output', help='Directory to save model and results')
    parser.add_argument('--epochs', type=int, default=100, help='Number of training epochs')
    parser.add_argument('--batch_size', type=int, default=32, help='Batch size for training')
    parser.add_argument('--lr', type=float, default=0.001, help='Learning rate')
    parser.add_argument('--hidden_dim', type=int, default=64, help='Hidden dimension size')
    parser.add_argument('--seed', type=int, default=42, help='Random seed')
    parser.add_argument('--eval_only', action='store_true', help='Run only evaluation on test set')
    parser.add_argument('--model_dir', type=str, help='Directory containing trained models for evaluation')
    parser.add_argument('--target', type=str, choices=['nodes', 'levels', 'iterations', 'all'], 
                        default='all', help='Target metric to evaluate')
    return parser.parse_args()

if __name__ == "__main__":
    args = parse_args()
    
    # Set random seed
    torch.manual_seed(args.seed)
    
    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Set device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Load dataset
    dataset = CircuitGraphDataset(args.csv_file, args.graph_dir)
    print(f"Dataset loaded with {len(dataset)} samples")
    
    # Get input dimensions from a sample
    sample = dataset[0]
    node_features = sample['graph_data'].x.shape[1]
    
    # Initialize model
    model = QoRPredictionModel(
        node_features=node_features,
        vocab_size=dataset.tokenizer.vocab_size,
        hidden_dim=193,
        output_dim=1
    )
    
    # Determine target metrics
    if args.target == 'all':
        target_metrics = ['nodes', 'levels', 'iterations']
    else:
        target_metrics = [args.target]
    
    # Filter target metrics based on what's available in the dataset
    available_targets = []
    for metric in target_metrics:
        if metric in sample['target']:
            available_targets.append(metric)
        else:
            print(f"Warning: Target metric '{metric}' not found in dataset")
    
    if not available_targets:
        print("Error: No valid target metrics found. Exiting.")
        exit(1)
    
    if args.eval_only:
        if not args.model_dir:
            args.model_dir = args.output_dir
        
        print(f"Evaluation mode: Using models from {args.model_dir}")
        evaluate_only(model, dataset, args, device, available_targets)
    else:
        # Train and evaluate model
        train_and_evaluate(model, dataset, args, device, available_targets)
