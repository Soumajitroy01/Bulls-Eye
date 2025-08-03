import os
import json
import torch
import networkx as nx
from bench_parser import BenchParser

def generate_graphs_from_folder(bench_folder, output_folder=None):
    """Generate PyG graph data from all bench files in a folder"""
    
    if not os.path.exists(bench_folder):
        raise FileNotFoundError(f"Bench folder {bench_folder} does not exist")
    
    if output_folder and not os.path.exists(output_folder):
        os.makedirs(output_folder)
    
    results = {}
    
    # Process each bench file
    for filename in os.listdir(bench_folder):
        if filename.endswith(".bench"):
            bench_path = os.path.join(bench_folder, filename)
            circuit_name = os.path.splitext(filename)[0]
            
            print(f"Processing {circuit_name}...")
            
            # Parse bench file
            parser = BenchParser(bench_path)
            data, graph_info = parser.to_pyg_data()
            
            # Save results
            results[circuit_name] = {
                'data': data,
                'graph_info': graph_info
            }
            
            # Save graph data if output folder is specified
            if output_folder:
                # Save PyG data
                torch.save(data, os.path.join(output_folder, f"{circuit_name}_pyg.pt"))
                
                # Save graph info as JSON
                with open(os.path.join(output_folder, f"{circuit_name}_info.json"), 'w') as f:
                    # Convert sets to lists for JSON serialization
                    serializable_info = {
                        'num_nodes': graph_info['num_nodes'],
                        'inputs': graph_info['inputs'],
                        'outputs': graph_info['outputs'],
                        'connections': graph_info['connections'],
                        'node_mapping': {str(k): v for k, v in graph_info['node_mapping'].items()},
                        'inverters': list(graph_info['inverters']) if 'inverters' in graph_info else []
                    }
                    json.dump(serializable_info, f, indent=2)
                
                # Create and save NetworkX graph for visualization purposes
                G = nx.DiGraph()
                                
                # Add nodes
                for node_id, node_name in graph_info['node_mapping'].items():
                    if node_name not in graph_info.get('inverters', set()):
                        G.add_node(node_id, name=node_name, 
                                is_input=(node_id in graph_info['inputs']),
                                is_output=(node_id in graph_info['outputs']))
                                
                # Add edges
                for src, dst, inverted in graph_info['connections']:
                    G.add_edge(src, dst, inverted=inverted)


                
                # Save NetworkX graph
                nx.write_gpickle(G, os.path.join(output_folder, f"{circuit_name}_nx.gpickle"))
                
                print(f"  - Saved graph data for {circuit_name}")
    
    print(f"Processed {len(results)} bench files")
    return results

def create_example_bench_file(output_path):
    """Create an example bench file for testing with NOT gates"""
    with open(output_path, 'w') as f:
        f.write("""# Simple AND-INV graph example with explicit NOT gates
INPUT(a)
INPUT(b)
INPUT(c)
INPUT(d)
OUTPUT(out1)
OUTPUT(out2)
n1 = AND(a, b)
not_c = NOT(c)
n2 = AND(not_c, d)
not_n2 = NOT(n2)
out1 = AND(n1, not_n2)
not_a = NOT(a)
out2 = AND(not_a, n2)
""")
    print(f"Created example bench file: {output_path}")

def main():
    # Example usage
    bench_folder = "bench_files"
    output_folder = "graph_data"
    
    # Create folders if they don't exist
    if not os.path.exists(bench_folder):
        os.makedirs(bench_folder)
        
    # Create an example bench file if the folder is empty
    if not any(f.endswith('.bench') for f in os.listdir(bench_folder)):
        create_example_bench_file(os.path.join(bench_folder, "example.bench"))
    
    # Generate graphs from bench files
    generate_graphs_from_folder(bench_folder, output_folder)
    
    print("Graph generation complete!")

if __name__ == "__main__":
    main()