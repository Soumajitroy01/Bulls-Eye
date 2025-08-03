import os
import argparse
from graph_generator import generate_graphs_from_folder, create_example_bench_file
from embedding_generator import generate_embeddings_from_folder

def main():
    parser = argparse.ArgumentParser(description='AIG to GCN Embeddings Pipeline')
    
    parser.add_argument('--bench_folder', type=str, default='bench_files',
                        help='Folder containing bench files')
    parser.add_argument('--graph_folder', type=str, default='graph_data',
                        help='Output folder for graph data')
    parser.add_argument('--embedding_folder', type=str, default='graph_embeddings',
                        help='Output folder for embeddings')
    parser.add_argument('--embedding_dim', type=int, default=128,
                        help='Dimension of output embeddings')
    parser.add_argument('--hidden_dim', type=int, default=128,
                        help='Dimension of hidden layers in GCN')
    parser.add_argument('--num_layers', type=int, default=3,
                        help='Number of GCN layers')
    parser.add_argument('--create_example', action='store_true',
                        help='Create example bench file if bench folder is empty')
    parser.add_argument('--skip_graph_generation', action='store_true',
                        help='Skip graph generation and use existing graph data')
    
    args = parser.parse_args()
    
    # Create folders if they don't exist
    for folder in [args.bench_folder, args.graph_folder, args.embedding_folder]:
        if not os.path.exists(folder):
            os.makedirs(folder)
    
    # Create example bench file if requested and folder is empty
    if args.create_example and not any(f.endswith('.bench') for f in os.listdir(args.bench_folder)):
        create_example_bench_file(os.path.join(args.bench_folder, "example.bench"))
    
    # Generate graphs if not skipped
    if not args.skip_graph_generation:
        print("\n=== Graph Generation ===")
        generate_graphs_from_folder(args.bench_folder, args.graph_folder)
    else:
        print("\n=== Skipping Graph Generation ===")
    
    # Generate embeddings
    print("\n=== Embedding Generation ===")
    generate_embeddings_from_folder(
        args.graph_folder, 
        args.embedding_folder,
        embedding_dim=args.embedding_dim
    )
    
    print("\n=== Pipeline Complete ===")
    print(f"Bench files processed from: {args.bench_folder}")
    print(f"Graph data saved to: {args.graph_folder}")
    print(f"Embeddings saved to: {args.embedding_folder}")

if __name__ == "__main__":
    main()