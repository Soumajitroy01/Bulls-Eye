import re
import torch
from torch_geometric.data import Data

class BenchParser:
    """Parser for bench files containing AIG descriptions with explicit NOT gates"""
    
    def __init__(self, bench_file_path):
        self.bench_file_path = bench_file_path
        self.nodes = {}  # Map node names to node ids
        self.inputs = []
        self.outputs = []
        self.connections = []  # List of tuples (source, target, is_inverted)
        self.inverters = set()  # Set to track nodes that are outputs of NOT gates
    
    def parse(self):
        """Parse the bench file and build the graph structure"""
        node_id = 0
        
        with open(self.bench_file_path, 'r') as f:
            lines = f.readlines()
        
        # First pass: identify all gates and assign node IDs
        for line in lines:
            line = line.strip()
            if not line or line.startswith('#'):
                continue
                
            # Handle input declaration
            if line.startswith('INPUT'):
                input_match = re.match(r'INPUT\((.*)\)', line)
                if input_match:
                    input_name = input_match.group(1)
                    self.nodes[input_name] = node_id
                    self.inputs.append(node_id)
                    node_id += 1
                    
            # Handle output declaration
            elif line.startswith('OUTPUT'):
                output_match = re.match(r'OUTPUT\((.*)\)', line)
                if output_match:
                    output_name = output_match.group(1)
                    # Output might reference an existing node
                    if output_name not in self.nodes:
                        self.nodes[output_name] = node_id
                        node_id += 1
                    self.outputs.append(self.nodes[output_name])
            
            # Handle assignments (gates)
            elif '=' in line:
                parts = line.split('=')
                target_name = parts[0].strip()
                
                # If node hasn't been encountered yet, assign an ID
                if target_name not in self.nodes:
                    self.nodes[target_name] = node_id
                    node_id += 1
        
        # Second pass: process gate connections
        for line in lines:
            line = line.strip()
            if not line or line.startswith('#') or line.startswith('INPUT') or line.startswith('OUTPUT'):
                continue
            
            if '=' in line:
                parts = line.split('=')
                target_name = parts[0].strip()
                gate_part = parts[1].strip()
                
                target_id = self.nodes[target_name]
                
                # Check if this is an AND gate
                if gate_part.startswith('AND'):
                    and_match = re.match(r'AND\((.*),(.*)\)', gate_part)
                    if and_match:
                        in1_name = and_match.group(1).strip()
                        in2_name = and_match.group(2).strip()
                        
                        # Check if inputs are outputs of NOT gates
                        in1_inv = in1_name in self.inverters
                        in2_inv = in2_name in self.inverters
                        
                        # Ensure nodes exist in our dictionary
                        if in1_name not in self.nodes:
                            self.nodes[in1_name] = node_id
                            node_id += 1
                        
                        if in2_name not in self.nodes:
                            self.nodes[in2_name] = node_id
                            node_id += 1
                        
                        # Add connections for AND gate
                        self.connections.append((self.nodes[in1_name], target_id, in1_inv))
                        self.connections.append((self.nodes[in2_name], target_id, in2_inv))
                
                # Check if this is a NOT gate
                elif gate_part.startswith('NOT'):
                    not_match = re.match(r'NOT\((.*)\)', gate_part)
                    if not_match:
                        in_name = not_match.group(1).strip()
                        
                        # Ensure node exists in our dictionary
                        if in_name not in self.nodes:
                            self.nodes[in_name] = node_id
                            node_id += 1
                        
                        # Add connection for NOT gate
                        # We mark this connection as inverted
                        self.connections.append((self.nodes[in_name], self.nodes[target_name], True))

                
                # Handle direct assignments (buffers)
                else:
                    source_name = gate_part
                    
                    # Ensure node exists in our dictionary
                    if source_name not in self.nodes:
                        self.nodes[source_name] = node_id
                        node_id += 1
                    
                    # Add connection for direct assignment (not inverted)
                    self.connections.append((self.nodes[source_name], target_id, False))
        
        return {
            'num_nodes': len(self.nodes),
            'inputs': self.inputs,
            'outputs': self.outputs,
            'connections': self.connections,
            'node_mapping': {v: k for k, v in self.nodes.items()},
            'inverters': self.inverters
        }

    def to_pyg_data(self):
        """Convert the parsed bench file to PyTorch Geometric Data object"""
        graph_info = self.parse()
        
        # Create edge index tensor
        edge_index = []
        edge_attr = []  # 1 for inverted, 0 for non-inverted
        
        for src, dst, inverted in graph_info['connections']:
            edge_index.append([src, dst])
            edge_attr.append([1.0 if inverted else 0.0])
        
        edge_index = torch.tensor(edge_index, dtype=torch.long).t().contiguous()
        edge_attr = torch.tensor(edge_attr, dtype=torch.float)
        
        # Create node features
        # For each node, we'll include:
        # - Is it an input? (1 or 0)
        # - Is it an output? (1 or 0)
        # - Is it an inverter output? (1 or 0)
        x = torch.zeros((graph_info['num_nodes'], 3), dtype=torch.float)
        
        for input_id in graph_info['inputs']:
            x[input_id, 0] = 1.0
            
        for output_id in graph_info['outputs']:
            x[output_id, 1] = 1.0
        
        # Mark inverter outputs
        for inv_name in graph_info['inverters']:
            if inv_name in self.nodes:
                inv_id = self.nodes[inv_name]
                x[inv_id, 2] = 1.0
        
        # Create PyG Data object
        data = Data(x=x, edge_index=edge_index, edge_attr=edge_attr)
        return data, graph_info