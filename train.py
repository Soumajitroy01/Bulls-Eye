import os
import torch
import torch.nn as nn
import numpy as np
from tqdm import tqdm
from torch.utils.data import DataLoader, SubsetRandomSampler
from sklearn.model_selection import train_test_split
from dataset import collate_fn
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

def train_model(model, train_loader, val_loader, criterion, optimizer, device, epochs, target_metric, args):
    history = {'train_loss': [], 'val_loss': []}
    best_val_loss = float('inf')
    best_train_loss = float('inf')
    
    for epoch in range(epochs):
        # Training
        model.train()
        train_loss = 0.0
        
        for batch in tqdm(train_loader, desc=f"Training epoch {epoch+1}"):
            graph_data = batch['graph_data'].to(device)
            recipe_tokens = batch['recipe_tokens'].to(device)
            target = batch['target'][target_metric].to(device) if target_metric in batch['target'] else None
            
            if target is not None:
                optimizer.zero_grad()
                output = model(graph_data, recipe_tokens)
                loss = criterion(output, target)
                loss.backward()
                optimizer.step()
                
                train_loss += loss.item() * graph_data.num_graphs
        
        train_loss /= len(train_loader.dataset)
        history['train_loss'].append(train_loss)
        
        # Validation
        model.eval()
        val_loss = 0.0
        
        with torch.no_grad():
            for batch in tqdm(val_loader, desc=f"Validating epoch {epoch+1}"):
                graph_data = batch['graph_data'].to(device)
                recipe_tokens = batch['recipe_tokens'].to(device)
                target = batch['target'][target_metric].to(device) if target_metric in batch['target'] else None
                
                if target is not None:
                    output = model(graph_data, recipe_tokens)
                    loss = criterion(output, target)
                    val_loss += loss.item() * graph_data.num_graphs
        
        val_loss /= len(val_loader.dataset)
        history['val_loss'].append(val_loss)
        
        print(f"Epoch {epoch+1}/{epochs}, Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}")

        # Save best model
        if train_loss < best_train_loss and val_loss < best_val_loss:
            best_val_loss = val_loss
            best_train_loss = train_loss
            torch.save(model.state_dict(), os.path.join(args.output_dir, f'best_model_9_{target_metric}.pt'))
            print(f"Saved new best model with train and loss: {train_loss:.4f} and val loss: {val_loss:.4f}")
        
        # Save best model
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save(model.state_dict(), os.path.join(args.output_dir, f'best_model_val_{target_metric}.pt'))
            print(f"Saved new best model with validation loss: {val_loss:.4f}")

        # Save best model
        if train_loss < best_train_loss:
            best_train_loss = train_loss
            torch.save(model.state_dict(), os.path.join(args.output_dir, f'best_model_train_{target_metric}.pt'))
            print(f"Saved new best model with train loss: {train_loss:.4f}")
        
        torch.save(model.state_dict(), os.path.join(args.output_dir, f'best_model_{epoch}_{target_metric}.pt'))
        print(f"Saved new epoch {epoch} model with train and loss: {train_loss:.4f} and val loss: {val_loss:.4f}")
    
    return history

def evaluate_model(model, test_loader, criterion, device, target_metric):
    model.eval()
    test_loss = 0.0
    predictions = []
    targets = []
    
    with torch.no_grad():
        for batch in tqdm(test_loader, desc=f"Testing model for {target_metric}"):
            graph_data = batch['graph_data'].to(device)
            recipe_tokens = batch['recipe_tokens'].to(device)
            target = batch['target'][target_metric].to(device) if target_metric in batch['target'] else None
            
            if target is not None:
                output = model(graph_data, recipe_tokens)
                loss = criterion(output, target)
                test_loss += loss.item() * graph_data.num_graphs
                
                predictions.extend(output.cpu().numpy())
                targets.extend(target.cpu().numpy())
    
    test_loss /= len(test_loader.dataset)
    
    # Calculate metrics
    mse = mean_squared_error(targets, predictions)
    mae = mean_absolute_error(targets, predictions)
    r2 = r2_score(targets, predictions)
    
    return test_loss, mse, mae, r2, predictions, targets

def plot_results(history, predictions, targets, target_metric, output_dir):
    # Plot training curves
    plt.figure(figsize=(10, 5))
    plt.plot(history['train_loss'], label='Train Loss')
    plt.plot(history['val_loss'], label='Validation Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title(f'Training and Validation Loss for {target_metric}')
    plt.legend()
    plt.grid(True)
    plt.savefig(os.path.join(output_dir, f'training_curves_{target_metric}.png'))
    
    # Plot predictions vs targets
    plt.figure(figsize=(10, 5))
    plt.scatter(targets, predictions, alpha=0.5)
    plt.plot([min(targets), max(targets)], [min(targets), max(targets)], 'r--')
    plt.xlabel('True Values')
    plt.ylabel('Predictions')
    plt.title(f'Predictions vs True Values for {target_metric}')
    plt.savefig(os.path.join(output_dir, f'predictions_{target_metric}.png'))

def evaluate_only(model, dataset, args, device, target_metrics):
    """Run only evaluation on the test set using pre-trained models."""
    from sklearn.model_selection import train_test_split
    import os
    import torch.nn as nn
    import pandas as pd
    
    # Split dataset - we only need the test set
    _, test_idx = train_test_split(range(len(dataset)), test_size=0.2, random_state=args.seed)
    test_sampler = SubsetRandomSampler(test_idx)
    
    # Create test data loader
    test_loader = DataLoader(
        dataset, batch_size=args.batch_size, sampler=test_sampler,
        collate_fn=collate_fn, num_workers=4
    )
    
    # Loss criterion
    criterion = nn.MSELoss()
    
    # Evaluate for each target metric
    for target_metric in target_metrics:
        model_path = os.path.join(args.model_dir, f'best_model_{target_metric}.pt')
        
        if not os.path.exists(model_path):
            print(f"Model file not found for {target_metric}: {model_path}")
            continue
        
        print(f"\nEvaluating model for target: {target_metric}")
        
        # Load model
        model.load_state_dict(torch.load(model_path, map_location=device))
        model = model.to(device)
        
        # Test model
        test_loss, mse, mae, r2, predictions, targets = evaluate_model(
            model=model,
            test_loader=test_loader,
            criterion=criterion,
            device=device,
            target_metric=target_metric
        )
        
        print(f"Test results for {target_metric}:")
        print(f"  Test Loss: {test_loss:.4f}")
        print(f"  MSE: {mse:.4f}")
        print(f"  MAE: {mae:.4f}")
        print(f"  R²: {r2:.4f}")
        
        # Plot predictions vs targets
        plt.figure(figsize=(10, 5))
        plt.scatter(targets, predictions, alpha=0.5)
        plt.plot([min(targets), max(targets)], [min(targets), max(targets)], 'r--')
        plt.xlabel('True Values')
        plt.ylabel('Predictions')
        plt.title(f'Predictions vs True Values for {target_metric}')
        plt.savefig(os.path.join(args.output_dir, f'eval_predictions_{target_metric}.png'))
        
        # Save predictions to CSV file
        results_df = pd.DataFrame({
            'True_Values': targets,
            'Predictions': [pred[0] for pred in predictions],  # Extract scalar values from prediction arrays
            'Absolute_Error': np.abs(np.array(targets) - np.array([pred[0] for pred in predictions]))
        })
        
        # Save to CSV
        csv_path = os.path.join(args.output_dir, f'predictions_{target_metric}.csv')
        results_df.to_csv(csv_path, index=False)
        print(f"Predictions saved to CSV: {csv_path}")
        
        # Save evaluation results
        results = {
            'test_loss': test_loss,
            'mse': mse,
            'mae': mae,
            'r2': r2,
            'predictions': predictions,
            'targets': targets
        }
        
        import pickle
        with open(os.path.join(args.output_dir, f'eval_results_{target_metric}.pkl'), 'wb') as f:
            pickle.dump(results, f)


def train_and_evaluate(model, dataset, args, device, target_metrics):
    # Split dataset
    train_idx, test_idx = train_test_split(range(len(dataset)), test_size=0.2, random_state=args.seed)
    train_idx, val_idx = train_test_split(train_idx, test_size=0.2, random_state=args.seed)
    
    # Create samplers
    train_sampler = SubsetRandomSampler(train_idx)
    val_sampler = SubsetRandomSampler(val_idx)
    test_sampler = SubsetRandomSampler(test_idx)
    
    # Create data loaders
    train_loader = DataLoader(
        dataset, batch_size=args.batch_size, sampler=train_sampler,
        collate_fn=collate_fn, num_workers=4
    )
    val_loader = DataLoader(
        dataset, batch_size=args.batch_size, sampler=val_sampler,
        collate_fn=collate_fn, num_workers=4
    )
    test_loader = DataLoader(
        dataset, batch_size=args.batch_size, sampler=test_sampler,
        collate_fn=collate_fn, num_workers=4
    )
    
    model = model.to(device)
    
    # Define loss function and optimizer
    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    
    # Train model for each target metric    
    for target_metric in target_metrics:
        if target_metric not in dataset[0]['target']:
            print(f"Skipping {target_metric} as it's not available in the dataset")
            continue
        
        print(f"\nTraining model for target: {target_metric}")
        
        # Reset model weights
        model.apply(lambda m: m.reset_parameters() if hasattr(m, 'reset_parameters') else None)
        
        # Train model
        history = train_model(
            model=model,
            train_loader=train_loader,
            val_loader=val_loader,
            criterion=criterion,
            optimizer=optimizer,
            device=device,
            epochs=args.epochs,
            target_metric=target_metric,
            args=args
        )
        
        # Load best model for testing
        model.load_state_dict(torch.load(os.path.join(args.output_dir, f'best_model_{target_metric}.pt')))
        
        # Test model
        test_loss, mse, mae, r2, predictions, targets = evaluate_model(
            model=model,
            test_loader=test_loader,
            criterion=criterion,
            device=device,
            target_metric=target_metric
        )
        
        print(f"Test results for {target_metric}:")
        print(f"  Test Loss: {test_loss:.4f}")
        print(f"  MSE: {mse:.4f}")
        print(f"  MAE: {mae:.4f}")
        print(f"  R²: {r2:.4f}")
        
        # Plot results
        plot_results(history, predictions, targets, target_metric, args.output_dir)
        
        # Save results
        results = {
            'test_loss': test_loss,
            'mse': mse,
            'mae': mae,
            'r2': r2,
            'predictions': predictions,
            'targets': targets,
            'history': history
        }
        
        import pickle
        with open(os.path.join(args.output_dir, f'results_{target_metric}.pkl'), 'wb') as f:
            pickle.dump(results, f)
