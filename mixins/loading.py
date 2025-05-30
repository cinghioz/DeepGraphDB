import dgl
import torch
import pandas as pd
import numpy as np
from typing import Dict, Tuple
import logging

logger = logging.getLogger(__name__)

class LoadingMixin:
    """Mixin for bulk loading operations"""

    def bulk_load_homogeneous_graph(self, edges_df: pd.DataFrame, nodes_df: pd.DataFrame = None):
        """
        Efficiently load a homogeneous graph from DataFrames.
        
        Args:
            edges_df: DataFrame with columns ['src', 'dst', ...metadata...]
            nodes_df: Optional DataFrame with node features ['node_id', ...metadata...]
        """
        # Validate edge indices
        max_node_id = max(edges_df['src'].max(), edges_df['dst'].max())
        if nodes_df is not None:
            max_node_id = max(max_node_id, nodes_df['node_id'].max())
        
        # Create edge tensors
        src_nodes = torch.tensor(edges_df['src'].values, dtype=torch.long)
        dst_nodes = torch.tensor(edges_df['dst'].values, dtype=torch.long)
        
        # Create DGL graph with explicit number of nodes
        self.graph = dgl.graph((src_nodes, dst_nodes), num_nodes=max_node_id + 1)
        
        # Add edge features/metadata
        edge_metadata_cols = [col for col in edges_df.columns if col not in ['src', 'dst']]
        for col in edge_metadata_cols:
            if edges_df[col].dtype in ['int64', 'float64', 'int32', 'float32']:
                self.graph.edata[col] = torch.tensor(edges_df[col].values, dtype=torch.float32)
            else:
                self.edge_data[col] = edges_df[col].values
        
        # Add node features/metadata if provided
        if nodes_df is not None:
            # Ensure all node IDs are present
            all_node_ids = range(max_node_id + 1)
            node_metadata_cols = [col for col in nodes_df.columns if col != 'node_id']
            
            for col in node_metadata_cols:
                if nodes_df[col].dtype in ['int64', 'float64', 'int32', 'float32']:
                    # Create tensor with proper ordering
                    node_values = nodes_df.set_index('node_id')[col].reindex(all_node_ids, fill_value=0)
                    self.graph.ndata[col] = torch.tensor(node_values.values, dtype=torch.float32)
                else:
                    self.node_data[col] = nodes_df[col].values
        
        print(f"Loaded homogeneous graph: {self.graph.num_nodes()} nodes, {self.graph.num_edges()} edges")
        return self.graph
    
    def bulk_load_heterogeneous_graph(self, node_types: Dict[str, pd.DataFrame], 
                                    edge_types: Dict[Tuple[str, str, str], pd.DataFrame]):
        """
        Efficiently load a heterogeneous graph with OPTIMIZED performance.
        
        Args:
            node_types: Dict mapping node_type -> DataFrame with node data
            edge_types: Dict mapping (src_type, edge_type, dst_type) -> DataFrame with edge data
        """
        import time
        start_time = time.time()
        
        print("Optimizing node mappings...")
        
        # Create fast lookup dictionaries for global_id -> local_id per type
        global_to_local = {}
        local_to_global = {}
        
        for node_type, nodes_df in node_types.items():
            # Vectorized mapping creation
            global_ids = nodes_df['node_id'].values
            local_ids = np.arange(len(global_ids))
        
            
            # Fast lookup dictionaries for edge processing
            global_to_local[node_type] = dict(zip(global_ids, local_ids))
            local_to_global[node_type] = dict(zip(local_ids, global_ids))
        
        print(f"Node mappings completed in {time.time() - start_time:.2f}s")
        
        # OPTIMIZATION 2: Process edges using vectorized pandas operations
        graph_data = {}
        edge_start = time.time()
        
        for (src_type, edge_type, dst_type), edges_df in edge_types.items():
            print(f"  Processing edge type {src_type}-{edge_type}->{dst_type} ({len(edges_df):,} edges)...")
            
            # FAST: Use pandas vectorized operations instead of loops
            src_mapping = global_to_local[src_type]
            dst_mapping = global_to_local[dst_type]
            
            # Vectorized ID conversion using pandas map (much faster than loops)
            edges_df_copy = edges_df.copy()
            edges_df_copy['src_local'] = edges_df_copy['src'].map(src_mapping)
            edges_df_copy['dst_local'] = edges_df_copy['dst'].map(dst_mapping)
            
            # Filter out edges with unmapped nodes (vectorized)
            valid_mask = (edges_df_copy['src_local'].notna()) & (edges_df_copy['dst_local'].notna())
            valid_edges = edges_df_copy[valid_mask]
            
            if len(valid_edges) > 0:
                # Convert to tensors (much faster than individual conversions)
                src_local = torch.tensor(valid_edges['src_local'].values, dtype=torch.long)
                dst_local = torch.tensor(valid_edges['dst_local'].values, dtype=torch.long)
                
                graph_data[(src_type, edge_type, dst_type)] = (src_local, dst_local)
                
                print(f"    Mapped {len(valid_edges):,} valid edges")
            else:
                print(f"    No valid edges found")
        
        print(f"Edge processing completed in {time.time() - edge_start:.2f}s")
        
        # OPTIMIZATION 3: Create graph with pre-computed node counts
        graph_start = time.time()
        num_nodes_dict = {ntype: len(nodes_df) for ntype, nodes_df in node_types.items()}
        self.graph = dgl.heterograph(graph_data, num_nodes_dict=num_nodes_dict)
        print(f"Graph creation completed in {time.time() - graph_start:.2f}s")
        
        # OPTIMIZATION 4: Add features efficiently using vectorized operations
        feature_start = time.time()
        
        # Add node features (vectorized)
        for node_type, nodes_df in node_types.items():
            metadata_cols = [col for col in nodes_df.columns if (col != 'node_id' and col != "node_type_id")]
            for col in metadata_cols:
                if nodes_df[col].dtype in ['int64', 'float64', 'int32', 'float32']:
                    # Direct tensor conversion (much faster)
                    self.graph.nodes[node_type].data[col] = torch.tensor(
                        nodes_df[col].values, dtype=torch.float32)
                else:
                    # Store non-numeric data separately
                    if node_type not in self.node_data:
                        self.node_data[node_type] = {}
                    self.node_data[node_type][col] = nodes_df[col].values
        
        # Add edge features (optimized)
        for (src_type, edge_type, dst_type), edges_df in edge_types.items():
            if (src_type, edge_type, dst_type) not in self.graph.canonical_etypes:
                continue
            
            metadata_cols = [col for col in edges_df.columns if col not in ['src', 'dst']]
            if not metadata_cols:
                continue
            
            # Get valid edges mapping (reuse previous computation)
            src_mapping = global_to_local[src_type]
            dst_mapping = global_to_local[dst_type]
            
            edges_df_copy = edges_df.copy()
            edges_df_copy['src_local'] = edges_df_copy['src'].map(src_mapping)
            edges_df_copy['dst_local'] = edges_df_copy['dst'].map(dst_mapping)
            valid_mask = (edges_df_copy['src_local'].notna()) & (edges_df_copy['dst_local'].notna())
            valid_edges = edges_df_copy[valid_mask]
            
            # Add features for valid edges only
            for col in metadata_cols:
                if valid_edges[col].dtype in ['int64', 'float64', 'int32', 'float32'] and col != "edge_type_id":
                    self.graph.edges[(src_type, edge_type, dst_type)].data[col] = torch.tensor(
                        valid_edges[col].values, dtype=torch.float32)
        
        print(f"Feature addition completed in {time.time() - feature_start:.2f}s")
        
        total_time = time.time() - start_time
        print(f"\nHeterogeneous graph loaded successfully in {total_time:.2f}s:")
        for ntype in self.graph.ntypes:
            print(f"  {ntype}: {self.graph.num_nodes(ntype):,} nodes")
        for etype in self.graph.canonical_etypes:
            print(f"  {etype}: {self.graph.num_edges(etype):,} edges")
        
        return self.graph
    