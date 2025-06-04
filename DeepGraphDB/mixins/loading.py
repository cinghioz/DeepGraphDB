import dgl
import torch
import pandas as pd
import numpy as np
from typing import Dict, Tuple, Any
import logging
import time

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

    def load_node_features_for_gnn(self, global_features_tensor: torch.Tensor, 
                               feature_name: str = "x") -> Dict[str, Any]:
        """
        Load node features for GNN training/inference from global feature tensor.
        
        Args:
            global_features_tensor: Tensor of shape [num_global_nodes, feat_dim]
                                Position i corresponds to global_id i
            feature_name: Name to store the features under (default: "gnn_features")
            
        Returns:
            Dict with loading statistics and feature organization info
        """
        if self.graph is None:
            raise ValueError("No graph loaded. Load a graph first.")
        
        if len(global_features_tensor.shape) != 2:
            raise ValueError(f"Expected 2D tensor [num_nodes, feat_dim], got shape {global_features_tensor.shape}")
        
        num_global_nodes, feat_dim = global_features_tensor.shape
        print(f"Loading {feat_dim}-dimensional features for {num_global_nodes} global nodes...")
        
        start_time = time.time()
        
        if not self.is_heterogeneous():
            # Homogeneous graph: simple case
            result = self._load_features_homogeneous(global_features_tensor, feature_name)
        else:
            # Heterogeneous graph: organize by node types
            result = self._load_features_heterogeneous(global_features_tensor, feature_name)
        
        loading_time = time.time() - start_time
        result['loading_time'] = loading_time
        result['feature_name'] = feature_name
        result['feature_dim'] = feat_dim
        
        print(f"Feature loading completed in {loading_time:.3f}s")
        print(f"Features organized by: {list(self.node_features.keys())}")
        
        return result

    def _load_features_homogeneous(self, global_features_tensor: torch.Tensor, 
                                feature_name: str) -> Dict[str, Any]:
        """Load features for homogeneous graphs."""
        num_nodes = self.graph.num_nodes()
        feat_dim = global_features_tensor.shape[1]
        
        # For homogeneous graphs, global_id == local_id, so we can use features directly
        # But we need to ensure we only take features for existing nodes
        max_global_id = min(global_features_tensor.shape[0] - 1, num_nodes - 1)
        
        if num_nodes <= global_features_tensor.shape[0]:
            # We have features for all nodes (and possibly more)
            self.node_features["nodes"] = global_features_tensor[:num_nodes].clone()
            used_features = num_nodes
        else:
            # We have fewer features than nodes - pad with zeros
            print(f"Warning: Only {global_features_tensor.shape[0]} features provided for {num_nodes} nodes")
            padded_features = torch.zeros(num_nodes, feat_dim, dtype=global_features_tensor.dtype)
            padded_features[:global_features_tensor.shape[0]] = global_features_tensor
            self.node_features["nodes"] = padded_features
            used_features = global_features_tensor.shape[0]
        
        # Also store in DGL graph for convenience
        self.graph.ndata[feature_name] = self.node_features["nodes"]
        
        return {
            'graph_type': 'homogeneous',
            'total_nodes': num_nodes,
            'features_used': used_features,
            'features_stored': {'nodes': self.node_features["nodes"].shape}
        }

    def _load_features_heterogeneous(self, global_features_tensor: torch.Tensor, 
                                    feature_name: str) -> Dict[str, Any]:
        """Load features for heterogeneous graphs organized by node type."""
        feat_dim = global_features_tensor.shape[1]
        features_by_type = {}
        stats = {}
        
        # Organize global_ids by node type and create ordered feature tensors
        for node_type in self.graph.ntypes:
            num_nodes_this_type = self.graph.num_nodes(node_type)
            
            # Collect global_ids for this node type in local_id order
            global_ids_ordered = []
            missing_global_ids = []
            
            for local_id in range(num_nodes_this_type):
                # Find the global_id that maps to (node_type, local_id)
                global_id = self.reverse_node_mapping.get((node_type, local_id))
                
                if global_id is not None:
                    global_ids_ordered.append(global_id)
                else:
                    # This shouldn't happen if mappings are correct
                    missing_global_ids.append(local_id)
                    global_ids_ordered.append(-1)  # Placeholder
            
            # Create feature tensor for this node type
            type_features = torch.zeros(num_nodes_this_type, feat_dim, 
                                    dtype=global_features_tensor.dtype)
            
            features_found = 0
            for local_id, global_id in enumerate(global_ids_ordered):
                if global_id != -1 and global_id < global_features_tensor.shape[0]:
                    type_features[local_id] = global_features_tensor[global_id]
                    features_found += 1
                # else: keep zeros for missing features
            
            # Store features
            self.node_features[node_type] = type_features
            features_by_type[node_type] = type_features
            
            # Also store in DGL graph
            self.graph.nodes[node_type].data[feature_name] = type_features
            
            # Statistics
            stats[node_type] = {
                'num_nodes': num_nodes_this_type,
                'features_found': features_found,
                'missing_global_ids': len(missing_global_ids),
                'feature_shape': tuple(type_features.shape)
            }
            
            if missing_global_ids:
                print(f"Warning: {len(missing_global_ids)} local_ids in {node_type} have no global_id mapping")
        
        total_nodes_with_features = sum(stats[nt]['features_found'] for nt in stats)
        total_nodes = sum(stats[nt]['num_nodes'] for nt in stats)
        
        return {
            'graph_type': 'heterogeneous',
            'node_types': list(self.graph.ntypes),
            'total_nodes': total_nodes,
            'total_features_assigned': total_nodes_with_features,
            'stats_by_type': stats,
            'coverage_ratio': total_nodes_with_features / total_nodes if total_nodes > 0 else 0
        }
    