import dgl
import torch
import pandas as pd
import numpy as np
from typing import Dict, List, Set, Tuple, Any, Optional, Union
import pickle
from collections import defaultdict, deque
import warnings
import time

class DGLGraphDatabase:
    """
    Scalable graph database using DGL for knowledge graphs and large-scale graph operations.
    Optimized for millions of nodes and edges with efficient bulk operations.
    """
    
    def __init__(self):
        self.graph = None
        self.node_data = {}
        self.edge_data = {}
        self.reverse_node_mapping = {}  # Maps (node_type, local_id) -> global_id
        self.global_to_local_mapping = {} # Maps global_id -> (node_type, local_id)
        self.node_types_mapping = {} # string -> int
        self.edge_types_mapping = {} # string -> int
        
    # =============================================================================
    # 1. BULK LOADING OF NODES AND EDGES
    # =============================================================================

    def set_mappings(self, node_types: Dict[str, int], edge_types: Dict[str, int]):
        self.node_types_mapping = node_types
        self.edge_types_mapping = edge_types

    def set_global_to_local_mapping(self, mapping: Dict[Tuple[str, int], int]):
        self.global_to_local_mapping = mapping
        self.reverse_node_mapping = dict(zip(mapping.values(), mapping.keys()))
    
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
    
    # =============================================================================
    # 2. BULK DELETION AND MODIFICATION #TODO
    # =============================================================================
    
    ##### NODES #####

    def bulk_add_nodes(self, nodes_data: Union[pd.DataFrame, Dict[str, List]], 
                   node_type: str = None) -> Dict[str, Any]:
        """
        Efficiently add new nodes in bulk to the graph.
        
        Args:
            nodes_data: DataFrame with node features or Dict mapping feature_name -> list of values
            node_type: For heterogeneous graphs, specify the node type. Required for heterogeneous graphs.
            
        Returns:
            Dict with addition statistics and new global IDs assigned
        """       
        print(f"Bulk adding nodes...")
        start_time = time.time()
        
        # Convert input to DataFrame if needed
        if isinstance(nodes_data, dict):
            nodes_df = pd.DataFrame(nodes_data)
        else:
            nodes_df = nodes_data.copy()
        
        num_new_nodes = len(nodes_df)
        if num_new_nodes == 0:
            return {'nodes_added': 0, 'new_global_ids': []}
        
        if not self.is_heterogeneous():
            result = self._bulk_add_nodes_homogeneous(nodes_df)
        else:
            if node_type is None:
                raise ValueError("node_type is required for heterogeneous graphs")
            result = self._bulk_add_nodes_heterogeneous(nodes_df, node_type)
        
        # Update ID mappings after adding nodes
        self._extend_id_mappings(result['new_global_ids'], result.get('node_type', 'default'), 
                                result['new_local_ids'])
        
        addition_time = time.time() - start_time
        result['addition_time'] = addition_time
        
        print(f"Node addition completed in {addition_time:.2f}s")
        print(f"Added {result['nodes_added']} nodes (Global IDs: {result['new_global_ids'][0]} to {result['new_global_ids'][-1]})")
        
        return result

    def _bulk_add_nodes_homogeneous(self, nodes_df: pd.DataFrame) -> Dict[str, Any]:
        """Handle bulk node addition for homogeneous graphs."""
        current_num_nodes = self.graph.num_nodes()
        num_new_nodes = len(nodes_df)
        new_total_nodes = current_num_nodes + num_new_nodes
        
        # Generate new global IDs (consecutive from current max)
        if len(self.global_to_local_mapping) > 0:
            next_global_id = max(self.global_to_local_mapping.keys()) + 1
        else:
            next_global_id = 0
        
        new_global_ids = list(range(next_global_id, next_global_id + num_new_nodes))
        new_local_ids = list(range(current_num_nodes, new_total_nodes))
        
        # Create new graph with additional nodes (preserve existing edges)
        if self.graph.num_edges() > 0:
            src, dst = self.graph.edges()
            new_graph = dgl.graph((src, dst), num_nodes=new_total_nodes)
        else:
            new_graph = dgl.graph((torch.tensor([]), torch.tensor([])), num_nodes=new_total_nodes)
        
        # Copy existing node features and extend with new data
        for feature_name in self.graph.ndata.keys():
            existing_features = self.graph.ndata[feature_name]
            
            if feature_name in nodes_df.columns:
                # Extend with new feature values
                new_values = nodes_df[feature_name].values
                if nodes_df[feature_name].dtype in ['int64', 'float64', 'int32', 'float32']:
                    new_tensor = torch.cat([
                        existing_features,
                        torch.tensor(new_values, dtype=torch.float32)
                    ])
                else:
                    # Non-numeric data goes to node_data
                    if feature_name not in self.node_data:
                        self.node_data[feature_name] = [''] * current_num_nodes
                    self.node_data[feature_name].extend(new_values.tolist())
                    new_tensor = torch.cat([
                        existing_features,
                        torch.zeros(num_new_nodes, dtype=existing_features.dtype)
                    ])
            else:
                # Feature not in new data, extend with zeros/defaults
                new_tensor = torch.cat([
                    existing_features,
                    torch.zeros(num_new_nodes, dtype=existing_features.dtype)
                ])
            
            new_graph.ndata[feature_name] = new_tensor
        
        # Add new features that don't exist in current graph
        for feature_name in nodes_df.columns:
            if feature_name not in self.graph.ndata:
                if nodes_df[feature_name].dtype in ['int64', 'float64', 'int32', 'float32']:
                    # Create new tensor feature
                    feature_tensor = torch.zeros(new_total_nodes, dtype=torch.float32)
                    feature_tensor[current_num_nodes:] = torch.tensor(
                        nodes_df[feature_name].values, dtype=torch.float32)
                    new_graph.ndata[feature_name] = feature_tensor
                else:
                    # Create new non-tensor feature
                    feature_list = [''] * current_num_nodes + nodes_df[feature_name].tolist()
                    self.node_data[feature_name] = feature_list
        
        # Copy existing edge features
        for feature_name, feature_tensor in self.graph.edata.items():
            new_graph.edata[feature_name] = feature_tensor
        
        self.graph = new_graph
        
        return {
            'nodes_added': num_new_nodes,
            'new_global_ids': new_global_ids,
            'new_local_ids': new_local_ids,
            'total_nodes': new_total_nodes
        }

    def _bulk_add_nodes_heterogeneous(self, nodes_df: pd.DataFrame, node_type: str) -> Dict[str, Any]:
        """Handle bulk node addition for heterogeneous graphs."""
        current_num_nodes = self.graph.num_nodes(node_type) if node_type in self.graph.ntypes else 0
        num_new_nodes = len(nodes_df)
        new_total_nodes = current_num_nodes + num_new_nodes
        
        # Generate new global IDs
        if len(self.global_to_local_mapping) > 0:
            next_global_id = max(self.global_to_local_mapping.keys()) + 1
        else:
            next_global_id = 0
        
        new_global_ids = list(range(next_global_id, next_global_id + num_new_nodes))
        new_local_ids = list(range(current_num_nodes, new_total_nodes))
        
        # Create new node count dictionary
        new_num_nodes_dict = {}
        for ntype in self.graph.ntypes:
            new_num_nodes_dict[ntype] = self.graph.num_nodes(ntype)
        new_num_nodes_dict[node_type] = new_total_nodes
        
        # Recreate graph with new node counts
        edge_dict = {}
        for etype in self.graph.canonical_etypes:
            src, dst = self.graph.edges(etype=etype)
            edge_dict[etype] = (src, dst)
        
        new_graph = dgl.heterograph(edge_dict, num_nodes_dict=new_num_nodes_dict)
        
        # Copy existing node features for all node types
        for ntype in self.graph.ntypes:
            for feature_name, feature_tensor in self.graph.nodes[ntype].data.items():
                new_graph.nodes[ntype].data[feature_name] = feature_tensor
        
        # Add/extend features for the target node type
        if node_type in self.graph.ntypes:
            # Extend existing features
            for feature_name in self.graph.nodes[node_type].data.keys():
                existing_features = self.graph.nodes[node_type].data[feature_name]
                
                if feature_name in nodes_df.columns:
                    new_values = nodes_df[feature_name].values
                    if nodes_df[feature_name].dtype in ['int64', 'float64', 'int32', 'float32']:
                        new_tensor = torch.cat([
                            existing_features,
                            torch.tensor(new_values, dtype=torch.float32)
                        ])
                    else:
                        # Handle non-numeric data
                        if node_type not in self.node_data:
                            self.node_data[node_type] = {}
                        if feature_name not in self.node_data[node_type]:
                            self.node_data[node_type][feature_name] = [''] * current_num_nodes
                        self.node_data[node_type][feature_name].extend(new_values.tolist())
                        new_tensor = torch.cat([
                            existing_features,
                            torch.zeros(num_new_nodes, dtype=existing_features.dtype)
                        ])
                else:
                    # Feature not in new data, extend with defaults
                    new_tensor = torch.cat([
                        existing_features,
                        torch.zeros(num_new_nodes, dtype=existing_features.dtype)
                    ])
                
                new_graph.nodes[node_type].data[feature_name] = new_tensor
        
        # Add new features that don't exist in current graph
        for feature_name in nodes_df.columns:
            if (node_type not in self.graph.ntypes or 
                feature_name not in self.graph.nodes[node_type].data):
                
                if nodes_df[feature_name].dtype in ['int64', 'float64', 'int32', 'float32']:
                    # Create new tensor feature
                    feature_tensor = torch.zeros(new_total_nodes, dtype=torch.float32)
                    feature_tensor[current_num_nodes:] = torch.tensor(
                        nodes_df[feature_name].values, dtype=torch.float32)
                    new_graph.nodes[node_type].data[feature_name] = feature_tensor
                else:
                    # Create new non-tensor feature
                    if node_type not in self.node_data:
                        self.node_data[node_type] = {}
                    feature_list = [''] * current_num_nodes + nodes_df[feature_name].tolist()
                    self.node_data[node_type][feature_name] = feature_list
        
        # Copy existing edge features
        for etype in self.graph.canonical_etypes:
            for feature_name, feature_tensor in self.graph.edges[etype].data.items():
                new_graph.edges[etype].data[feature_name] = feature_tensor
        
        self.graph = new_graph
        
        return {
            'nodes_added': num_new_nodes,
            'new_global_ids': new_global_ids,
            'new_local_ids': new_local_ids,
            'node_type': node_type,
            'total_nodes': new_total_nodes
        }

    def _extend_id_mappings(self, new_global_ids: List[int], node_type: str, new_local_ids: List[int]):
        """Extend ID mappings with new nodes."""
        for global_id, local_id in zip(new_global_ids, new_local_ids):
            self.global_to_local_mapping[global_id] = (node_type, local_id)
            self.reverse_node_mapping[(node_type, local_id)] = global_id

    def bulk_delete_nodes(self, global_ids_to_delete: List[int], 
                     preserve_edges: bool = False) -> Dict[str, Any]:
        """
        Efficiently delete nodes in bulk and recompute all ID mappings.
        
        Args:
            global_ids_to_delete: List of global node IDs to delete
            preserve_edges: If True, keep edges between remaining nodes, else remove all connected edges
            
        Returns:
            Dict with deletion statistics and new mappings
        """
        if self.graph is None:
            raise ValueError("No graph loaded")
        
        if not global_ids_to_delete:
            return {'nodes_deleted': 0, 'edges_deleted': 0}
        
        print(f"Bulk deleting {len(global_ids_to_delete)} nodes...")
        start_time = time.time()
        
        delete_set = set(global_ids_to_delete)
        
        if not self.is_heterogeneous():
            result = self._bulk_delete_nodes_homogeneous(delete_set, preserve_edges)
        else:
            result = self._bulk_delete_nodes_heterogeneous(delete_set, preserve_edges)
        
        # Recompute all ID mappings with consecutive numbering
        self._recompute_id_mappings()
        
        deletion_time = time.time() - start_time
        result['deletion_time'] = deletion_time
        
        print(f"Deletion completed in {deletion_time:.2f}s")
        print(f"Deleted {result['nodes_deleted']} nodes and {result['edges_deleted']} edges")
        
        return result

    def _bulk_delete_nodes_homogeneous(self, delete_set: Set[int], preserve_edges: bool) -> Dict[str, Any]:
        """Handle bulk deletion for homogeneous graphs."""
        original_num_nodes = self.graph.num_nodes()
        original_num_edges = self.graph.num_edges()
        
        # Create mapping from old to new node IDs
        remaining_nodes = [i for i in range(original_num_nodes) if i not in delete_set]
        
        if not remaining_nodes:
            # All nodes deleted - create empty graph
            self.graph = dgl.graph((torch.tensor([]), torch.tensor([])), num_nodes=0)
            self.node_data = {}
            self.edge_data = {}
            return {
                'nodes_deleted': original_num_nodes,
                'edges_deleted': original_num_edges,
                'remaining_nodes': 0
            }
        
        # Create old_to_new mapping
        old_to_new = {old_id: new_id for new_id, old_id in enumerate(remaining_nodes)}
        
        # Get current edges and filter out edges connected to deleted nodes
        src, dst = self.graph.edges()
        src_list = src.tolist()
        dst_list = dst.tolist()
        
        valid_edges = []
        for i, (s, d) in enumerate(zip(src_list, dst_list)):
            if s not in delete_set and d not in delete_set:
                new_s = old_to_new[s]
                new_d = old_to_new[d]
                valid_edges.append((new_s, new_d, i))  # Include original edge index
        
        # Create new graph
        if valid_edges:
            new_src, new_dst, original_edge_indices = zip(*valid_edges)
            new_src_tensor = torch.tensor(new_src, dtype=torch.long)
            new_dst_tensor = torch.tensor(new_dst, dtype=torch.long)
        else:
            new_src_tensor = torch.tensor([], dtype=torch.long)
            new_dst_tensor = torch.tensor([], dtype=torch.long)
            original_edge_indices = []
        
        self.graph = dgl.graph((new_src_tensor, new_dst_tensor), num_nodes=len(remaining_nodes))
        
        # Update node features
        remaining_node_tensor = torch.tensor(remaining_nodes, dtype=torch.long)
        for feature_name, feature_tensor in self.graph.ndata.items():
            self.graph.ndata[feature_name] = feature_tensor[remaining_node_tensor]
        
        # Update edge features
        if original_edge_indices:
            edge_index_tensor = torch.tensor(original_edge_indices, dtype=torch.long)
            for feature_name, feature_tensor in self.graph.edata.items():
                self.graph.edata[feature_name] = feature_tensor[edge_index_tensor]
        
        # Update non-tensor node data
        for feature_name, feature_data in self.node_data.items():
            if isinstance(feature_data, np.ndarray):
                self.node_data[feature_name] = feature_data[remaining_nodes]
            elif isinstance(feature_data, list):
                self.node_data[feature_name] = [feature_data[i] for i in remaining_nodes]
        
        # Update non-tensor edge data
        if original_edge_indices:
            for feature_name, feature_data in self.edge_data.items():
                if isinstance(feature_data, np.ndarray):
                    self.edge_data[feature_name] = feature_data[list(original_edge_indices)]
                elif isinstance(feature_data, list):
                    self.edge_data[feature_name] = [feature_data[i] for i in original_edge_indices]
        
        return {
            'nodes_deleted': len(delete_set),
            'edges_deleted': original_num_edges - len(original_edge_indices),
            'remaining_nodes': len(remaining_nodes)
        }

    def _bulk_delete_nodes_heterogeneous(self, delete_set: Set[int], preserve_edges: bool) -> Dict[str, Any]:
        """Handle bulk deletion for heterogeneous graphs."""
        # Group nodes to delete by type
        nodes_to_delete_by_type = defaultdict(list)
        
        for global_id in delete_set:
            if global_id in self.global_to_local_mapping:
                node_type, local_id = self.global_to_local_mapping[global_id]
                nodes_to_delete_by_type[node_type].append(local_id)
        
        # Create new node data for each type
        new_node_data = {}
        new_node_counts = {}
        node_mappings = {}  # old_local_id -> new_local_id per type
        
        for node_type in self.graph.ntypes:
            total_nodes = self.graph.num_nodes(node_type)
            delete_locals = set(nodes_to_delete_by_type.get(node_type, []))
            
            # Keep nodes not in delete set
            remaining_locals = [i for i in range(total_nodes) if i not in delete_locals]
            
            new_node_counts[node_type] = len(remaining_locals)
            node_mappings[node_type] = {old_id: new_id for new_id, old_id in enumerate(remaining_locals)}
            
            # Update node features for this type
            if remaining_locals:
                remaining_tensor = torch.tensor(remaining_locals, dtype=torch.long)
                
                # Update DGL node data
                for feature_name, feature_tensor in self.graph.nodes[node_type].data.items():
                    if node_type not in new_node_data:
                        new_node_data[node_type] = {}
                    new_node_data[node_type][feature_name] = feature_tensor[remaining_tensor]
                
                # Update non-tensor node data
                if node_type in self.node_data:
                    for feature_name, feature_data in self.node_data[node_type].items():
                        if node_type not in new_node_data:
                            new_node_data[node_type] = {}
                        
                        if isinstance(feature_data, np.ndarray):
                            new_node_data[node_type][feature_name] = feature_data[remaining_locals]
                        elif isinstance(feature_data, list):
                            new_node_data[node_type][feature_name] = [feature_data[i] for i in remaining_locals]
        
        # Create new edge data
        new_edge_data = {}
        total_edges_deleted = 0
        
        for src_type, edge_type, dst_type in self.graph.canonical_etypes:
            if (new_node_counts.get(src_type, 0) == 0 or 
                new_node_counts.get(dst_type, 0) == 0):
                # Skip edge types where source or destination type has no remaining nodes
                total_edges_deleted += self.graph.num_edges((src_type, edge_type, dst_type))
                continue
            
            # Get current edges
            src, dst = self.graph.edges(etype=(src_type, edge_type, dst_type))
            
            # Filter valid edges and remap IDs
            valid_edges = []
            original_edge_indices = []
            
            for i, (s, d) in enumerate(zip(src.tolist(), dst.tolist())):
                if (s in node_mappings[src_type] and d in node_mappings[dst_type]):
                    new_s = node_mappings[src_type][s]
                    new_d = node_mappings[dst_type][d]
                    valid_edges.append((new_s, new_d))
                    original_edge_indices.append(i)
            
            if valid_edges:
                new_src, new_dst = zip(*valid_edges)
                new_edge_data[(src_type, edge_type, dst_type)] = (
                    torch.tensor(new_src, dtype=torch.long),
                    torch.tensor(new_dst, dtype=torch.long)
                )
                
                # Store edge features
                if original_edge_indices:
                    edge_idx_tensor = torch.tensor(original_edge_indices, dtype=torch.long)
                    edge_features = {}
                    
                    # DGL edge features
                    for feat_name, feat_tensor in self.graph.edges[(src_type, edge_type, dst_type)].data.items():
                        edge_features[feat_name] = feat_tensor[edge_idx_tensor]
                    
                    new_edge_data[(src_type, edge_type, dst_type)] = (
                        new_edge_data[(src_type, edge_type, dst_type)][0],  # src
                        new_edge_data[(src_type, edge_type, dst_type)][1],  # dst
                        edge_features  # features
                    )
            
            # Count deleted edges
            original_edge_count = self.graph.num_edges((src_type, edge_type, dst_type))
            remaining_edge_count = len(valid_edges) if valid_edges else 0
            total_edges_deleted += original_edge_count - remaining_edge_count
        
        # Create new heterogeneous graph
        graph_data_dict = {}
        for etype, edge_info in new_edge_data.items():
            graph_data_dict[etype] = (edge_info[0], edge_info[1])
        
        self.graph = dgl.heterograph(graph_data_dict, num_nodes_dict=new_node_counts)
        
        # Set node features
        for node_type, features in new_node_data.items():
            for feature_name, feature_tensor in features.items():
                if isinstance(feature_tensor, torch.Tensor):
                    self.graph.nodes[node_type].data[feature_name] = feature_tensor
        
        # Set edge features
        for etype, edge_info in new_edge_data.items():
            if len(edge_info) > 2:  # Has features
                edge_features = edge_info[2]
                for feature_name, feature_tensor in edge_features.items():
                    self.graph.edges[etype].data[feature_name] = feature_tensor
        
        # Update non-tensor data storage
        self.node_data = {node_type: {feat: val for feat, val in features.items() 
                                    if not isinstance(val, torch.Tensor)}
                        for node_type, features in new_node_data.items()}
        
        total_nodes_deleted = len(delete_set)
        remaining_nodes = sum(new_node_counts.values())
        
        return {
            'nodes_deleted': total_nodes_deleted,
            'edges_deleted': total_edges_deleted,
            'remaining_nodes': remaining_nodes,
            'nodes_by_type': new_node_counts
        }

    def _recompute_id_mappings(self):
        """Recompute global and local ID mappings with consecutive numbering."""
        self.global_to_local_mapping = {}
        self.reverse_node_mapping = {}
        
        global_id_counter = 0
        
        if not self.is_heterogeneous():
            # Homogeneous graph: simple consecutive mapping
            for local_id in range(self.graph.num_nodes()):
                self.global_to_local_mapping[global_id_counter] = ('default', local_id)
                self.reverse_node_mapping[('default', local_id)] = global_id_counter
                global_id_counter += 1
        else:
            # Heterogeneous graph: consecutive numbering across all types
            for node_type in self.graph.ntypes:
                for local_id in range(self.graph.num_nodes(node_type)):
                    self.global_to_local_mapping[global_id_counter] = (node_type, local_id)
                    self.reverse_node_mapping[(node_type, local_id)] = global_id_counter
                    global_id_counter += 1
        
        print(f"ID mappings recomputed: {global_id_counter} total nodes")
    
    def bulk_modify_nodes(self, modifications: Dict[int, Dict[str, Any]], 
                     modify_edges: bool = False) -> Dict[str, Any]:
        """
        Efficiently modify node (and optionally edge) attributes in bulk.
        
        Args:
            modifications: Dict mapping global_id -> {feature_name: new_value}
            modify_edges: If True, also modify edge attributes for edges connected to these nodes
            
        Returns:
            Dict with modification statistics
        """
        if self.graph is None:
            raise ValueError("No graph loaded")
        
        if not modifications:
            return {'nodes_modified': 0, 'features_modified': 0}
        
        print(f"Bulk modifying {len(modifications)} nodes...")
        start_time = time.time()
        
        if not self.is_heterogeneous():
            result = self._bulk_modify_homogeneous(modifications, modify_edges)
        else:
            result = self._bulk_modify_heterogeneous(modifications, modify_edges)
        
        modification_time = time.time() - start_time
        result['modification_time'] = modification_time
        
        print(f"Modification completed in {modification_time:.2f}s")
        print(f"Modified {result['nodes_modified']} nodes, {result['features_modified']} features")
        
        return result

    def _bulk_modify_homogeneous(self, modifications: Dict[int, Dict[str, Any]], 
                            modify_edges: bool) -> Dict[str, Any]:
        """Handle bulk modification for homogeneous graphs."""
        nodes_modified = 0
        features_modified = 0
        
        # Group modifications by feature for efficient batch updates
        feature_updates = defaultdict(dict)  # feature_name -> {node_id: new_value}
        
        for global_id, node_updates in modifications.items():
            if global_id >= self.graph.num_nodes():
                print(f"Warning: Node {global_id} not found, skipping")
                continue
            
            nodes_modified += 1
            for feature_name, new_value in node_updates.items():
                feature_updates[feature_name][global_id] = new_value
                features_modified += 1
        
        # Apply updates efficiently
        for feature_name, updates in feature_updates.items():
            if feature_name in self.graph.ndata:
                # Update DGL tensor features
                for node_id, new_value in updates.items():
                    if isinstance(new_value, (int, float)):
                        self.graph.ndata[feature_name][node_id] = new_value
                    elif isinstance(new_value, torch.Tensor):
                        self.graph.ndata[feature_name][node_id] = new_value
                    else:
                        # Convert to tensor if possible
                        try:
                            self.graph.ndata[feature_name][node_id] = torch.tensor(new_value)
                        except:
                            print(f"Warning: Cannot convert {new_value} to tensor for feature {feature_name}")
            
            elif feature_name in self.node_data:
                # Update non-tensor features
                for node_id, new_value in updates.items():
                    if isinstance(self.node_data[feature_name], np.ndarray):
                        self.node_data[feature_name][node_id] = new_value
                    elif isinstance(self.node_data[feature_name], list):
                        if node_id < len(self.node_data[feature_name]):
                            self.node_data[feature_name][node_id] = new_value
            else:
                # Create new feature
                if all(isinstance(v, (int, float)) for v in updates.values()):
                    # Create tensor feature
                    feature_tensor = torch.zeros(self.graph.num_nodes(), dtype=torch.float32)
                    for node_id, new_value in updates.items():
                        feature_tensor[node_id] = new_value
                    self.graph.ndata[feature_name] = feature_tensor
                else:
                    # Create non-tensor feature
                    feature_list = [None] * self.graph.num_nodes()
                    for node_id, new_value in updates.items():
                        feature_list[node_id] = new_value
                    self.node_data[feature_name] = feature_list
        
        return {
            'nodes_modified': nodes_modified,
            'features_modified': features_modified,
            'feature_names': list(feature_updates.keys())
        }

    def _bulk_modify_heterogeneous(self, modifications: Dict[int, Dict[str, Any]], 
                                modify_edges: bool) -> Dict[str, Any]:
        """Handle bulk modification for heterogeneous graphs."""
        nodes_modified = 0
        features_modified = 0
        modifications_by_type = defaultdict(dict)  # node_type -> {local_id: {feature: value}}
        
        # Group modifications by node type
        for global_id, node_updates in modifications.items():
            if global_id not in self.global_to_local_mapping:
                print(f"Warning: Global ID {global_id} not found in mapping, skipping")
                continue
            
            node_type, local_id = self.global_to_local_mapping[global_id]
            modifications_by_type[node_type][local_id] = node_updates
            nodes_modified += 1
            features_modified += len(node_updates)
        
        # Apply modifications for each node type
        for node_type, type_modifications in modifications_by_type.items():
            # Group by feature for efficient updates
            feature_updates = defaultdict(dict)  # feature_name -> {local_id: new_value}
            
            for local_id, node_updates in type_modifications.items():
                for feature_name, new_value in node_updates.items():
                    feature_updates[feature_name][local_id] = new_value
            
            # Apply feature updates
            for feature_name, updates in feature_updates.items():
                if feature_name in self.graph.nodes[node_type].data:
                    # Update DGL tensor features
                    for local_id, new_value in updates.items():
                        if isinstance(new_value, (int, float)):
                            self.graph.nodes[node_type].data[feature_name][local_id] = new_value
                        elif isinstance(new_value, torch.Tensor):
                            self.graph.nodes[node_type].data[feature_name][local_id] = new_value
                        else:
                            try:
                                self.graph.nodes[node_type].data[feature_name][local_id] = torch.tensor(new_value)
                            except:
                                print(f"Warning: Cannot convert {new_value} to tensor for {node_type}.{feature_name}")
                
                elif node_type in self.node_data and feature_name in self.node_data[node_type]:
                    # Update non-tensor features
                    for local_id, new_value in updates.items():
                        if isinstance(self.node_data[node_type][feature_name], np.ndarray):
                            self.node_data[node_type][feature_name][local_id] = new_value
                        elif isinstance(self.node_data[node_type][feature_name], list):
                            if local_id < len(self.node_data[node_type][feature_name]):
                                self.node_data[node_type][feature_name][local_id] = new_value
                else:
                    # Create new feature for this node type
                    if all(isinstance(v, (int, float)) for v in updates.values()):
                        # Create tensor feature
                        feature_tensor = torch.zeros(self.graph.num_nodes(node_type), dtype=torch.float32)
                        for local_id, new_value in updates.items():
                            feature_tensor[local_id] = new_value
                        self.graph.nodes[node_type].data[feature_name] = feature_tensor
                    else:
                        # Create non-tensor feature
                        if node_type not in self.node_data:
                            self.node_data[node_type] = {}
                        feature_list = [None] * self.graph.num_nodes(node_type)
                        for local_id, new_value in updates.items():
                            feature_list[local_id] = new_value
                        self.node_data[node_type][feature_name] = feature_list
        
        return {
            'nodes_modified': nodes_modified,
            'features_modified': features_modified,
            'modifications_by_type': {ntype: len(mods) for ntype, mods in modifications_by_type.items()}
        }
    
    ##### EDGES #####
    
    def bulk_add_edges(self, edges_data: Union[pd.DataFrame, Dict[str, List]], 
                   edge_type: Tuple[str, str, str] = None) -> Dict[str, Any]:
        """
        Efficiently add new edges in bulk to the graph between existing nodes.
        
        Args:
            edges_data: DataFrame with columns ['src', 'dst', ...features...] or Dict with edge data
                    'src' and 'dst' must contain global node IDs that already exist in the graph
            edge_type: For heterogeneous graphs, specify (src_type, edge_rel, dst_type)
            
        Returns:
            Dict with addition statistics
        """
        if self.graph is None:
            raise ValueError("Cannot add edges to non-existent graph. Add nodes first.")
        
        print(f"Bulk adding edges...")
        start_time = time.time()
        
        # Convert input to DataFrame if needed
        if isinstance(edges_data, dict):
            edges_df = pd.DataFrame(edges_data)
        else:
            edges_df = edges_data.copy()
        
        if len(edges_df) == 0:
            return {'edges_added': 0, 'invalid_edges': 0}
        
        # Validate required columns
        if 'src' not in edges_df.columns or 'dst' not in edges_df.columns:
            raise ValueError("edges_data must contain 'src' and 'dst' columns")
        
        # Validate that all nodes exist
        validation_result = self._validate_edge_nodes(edges_df, edge_type)
        if validation_result['invalid_edges'] > 0:
            print(f"Warning: {validation_result['invalid_edges']} edges reference non-existent nodes and will be skipped")
            edges_df = validation_result['valid_edges_df']
        
        if len(edges_df) == 0:
            return {'edges_added': 0, 'invalid_edges': validation_result['invalid_edges']}
        
        # Add the edges
        if not self.is_heterogeneous():
            edge_result = self._bulk_add_edges_homogeneous(edges_df)
        else:
            if edge_type is None:
                raise ValueError("edge_type is required for heterogeneous graphs")
            edge_result = self._bulk_add_edges_heterogeneous(edges_df, edge_type)
        
        edge_result['invalid_edges'] = validation_result['invalid_edges']
        
        addition_time = time.time() - start_time
        edge_result['addition_time'] = addition_time
        
        print(f"Edge addition completed in {addition_time:.2f}s")
        print(f"Added {edge_result['edges_added']} edges")
        
        return edge_result

    def _validate_edge_nodes(self, edges_df: pd.DataFrame, 
                            edge_type: Tuple[str, str, str] = None) -> Dict[str, Any]:
        """
        Validate that all nodes referenced in edges exist in the graph.
        
        Args:
            edges_df: DataFrame with src/dst columns containing global node IDs
            edge_type: For heterogeneous graphs, the edge type to validate against
            
        Returns:
            Dict with validation results and filtered DataFrame
        """
        existing_global_ids = set(self.global_to_local_mapping.keys())
        valid_edge_indices = []
        invalid_count = 0
        
        if not self.is_heterogeneous():
            # Homogeneous graph - just check if global IDs exist
            for i, (src, dst) in enumerate(zip(edges_df['src'], edges_df['dst'])):
                if src in existing_global_ids and dst in existing_global_ids:
                    valid_edge_indices.append(i)
                else:
                    invalid_count += 1
                    missing_nodes = []
                    if src not in existing_global_ids:
                        missing_nodes.append(f"src:{src}")
                    if dst not in existing_global_ids:
                        missing_nodes.append(f"dst:{dst}")
                    print(f"Warning: Edge ({src}, {dst}) references non-existent nodes: {missing_nodes}")
        
        else:
            # Heterogeneous graph - check both existence and node types
            if edge_type is None:
                raise ValueError("edge_type is required for heterogeneous graph validation")
            
            src_type, edge_rel, dst_type = edge_type
            
            for i, (src, dst) in enumerate(zip(edges_df['src'], edges_df['dst'])):
                src_valid = dst_valid = False
                
                # Check if source node exists and has correct type
                if src in self.global_to_local_mapping:
                    actual_src_type, _ = self.global_to_local_mapping[src]
                    if actual_src_type == src_type:
                        src_valid = True
                    else:
                        print(f"Warning: Source node {src} has type '{actual_src_type}', expected '{src_type}'")
                else:
                    print(f"Warning: Source node {src} does not exist")
                
                # Check if destination node exists and has correct type
                if dst in self.global_to_local_mapping:
                    actual_dst_type, _ = self.global_to_local_mapping[dst]
                    if actual_dst_type == dst_type:
                        dst_valid = True
                    else:
                        print(f"Warning: Destination node {dst} has type '{actual_dst_type}', expected '{dst_type}'")
                else:
                    print(f"Warning: Destination node {dst} does not exist")
                
                if src_valid and dst_valid:
                    valid_edge_indices.append(i)
                else:
                    invalid_count += 1
        
        # Create filtered DataFrame with only valid edges
        if valid_edge_indices:
            valid_edges_df = edges_df.iloc[valid_edge_indices].copy()
        else:
            valid_edges_df = pd.DataFrame(columns=edges_df.columns)
        
        return {
            'valid_edges_df': valid_edges_df,
            'invalid_edges': invalid_count,
            'valid_edges': len(valid_edge_indices)
        }

    def _bulk_add_edges_homogeneous(self, edges_df: pd.DataFrame) -> Dict[str, Any]:
        """Handle bulk edge addition for homogeneous graphs."""
        current_edges = self.graph.num_edges()
        
        # Get existing edges
        if current_edges > 0:
            existing_src, existing_dst = self.graph.edges()
            existing_src = existing_src.tolist()
            existing_dst = existing_dst.tolist()
        else:
            existing_src, existing_dst = [], []
        
        # Add new edges
        new_src = edges_df['src'].tolist()
        new_dst = edges_df['dst'].tolist()
        
        # Combine all edges
        all_src = existing_src + new_src
        all_dst = existing_dst + new_dst
        
        # Create new graph with all edges
        new_graph = dgl.graph((torch.tensor(all_src), torch.tensor(all_dst)), 
                            num_nodes=self.graph.num_nodes())
        
        # Copy existing node features
        for feature_name, feature_tensor in self.graph.ndata.items():
            new_graph.ndata[feature_name] = feature_tensor
        
        # Handle edge features
        edge_metadata_cols = [col for col in edges_df.columns if col not in ['src', 'dst']]
        
        # Copy existing edge features and extend with new data
        for feature_name in self.graph.edata.keys():
            existing_features = self.graph.edata[feature_name]
            
            if feature_name in edge_metadata_cols:
                # Extend with new feature values
                new_values = edges_df[feature_name].values
                if edges_df[feature_name].dtype in ['int64', 'float64', 'int32', 'float32']:
                    new_tensor = torch.cat([
                        existing_features,
                        torch.tensor(new_values, dtype=torch.float32)
                    ])
                else:
                    # Non-numeric data
                    if feature_name not in self.edge_data:
                        self.edge_data[feature_name] = [''] * current_edges
                    self.edge_data[feature_name].extend(new_values.tolist())
                    new_tensor = torch.cat([
                        existing_features,
                        torch.zeros(len(edges_df), dtype=existing_features.dtype)
                    ])
            else:
                # Feature not in new data, extend with defaults
                new_tensor = torch.cat([
                    existing_features,
                    torch.zeros(len(edges_df), dtype=existing_features.dtype)
                ])
            
            new_graph.edata[feature_name] = new_tensor
        
        # Add new features that don't exist in current graph
        for feature_name in edge_metadata_cols:
            if feature_name not in self.graph.edata:
                if edges_df[feature_name].dtype in ['int64', 'float64', 'int32', 'float32']:
                    # Create new tensor feature
                    feature_tensor = torch.zeros(current_edges + len(edges_df), dtype=torch.float32)
                    feature_tensor[current_edges:] = torch.tensor(
                        edges_df[feature_name].values, dtype=torch.float32)
                    new_graph.edata[feature_name] = feature_tensor
                else:
                    # Create new non-tensor feature
                    feature_list = [''] * current_edges + edges_df[feature_name].tolist()
                    self.edge_data[feature_name] = feature_list
        
        self.graph = new_graph
        
        return {
            'edges_added': len(edges_df),
            'total_edges': current_edges + len(edges_df)
        }

    def _bulk_add_edges_heterogeneous(self, edges_df: pd.DataFrame, 
                                    edge_type: Tuple[str, str, str]) -> Dict[str, Any]:
        """Handle bulk edge addition for heterogeneous graphs."""
        src_type, edge_rel, dst_type = edge_type
        
        # Convert global IDs to local IDs
        src_local = []
        dst_local = []
        valid_edges = []
        
        for i, (src_global, dst_global) in enumerate(zip(edges_df['src'], edges_df['dst'])):
            if (src_global in self.global_to_local_mapping and 
                dst_global in self.global_to_local_mapping):
                
                src_node_type, src_local_id = self.global_to_local_mapping[src_global]
                dst_node_type, dst_local_id = self.global_to_local_mapping[dst_global]
                
                if src_node_type == src_type and dst_node_type == dst_type:
                    src_local.append(src_local_id)
                    dst_local.append(dst_local_id)
                    valid_edges.append(i)
                else:
                    print(f"Warning: Node type mismatch for edge ({src_global}, {dst_global})")
            else:
                print(f"Warning: Global ID not found for edge ({src_global}, {dst_global})")
        
        if not valid_edges:
            return {'edges_added': 0, 'total_edges': self.graph.num_edges()}
        
        # Get existing edges for this edge type
        edge_dict = {}
        for etype in self.graph.canonical_etypes:
            existing_src, existing_dst = self.graph.edges(etype=etype)
            edge_dict[etype] = (existing_src, existing_dst)
        
        # Add new edges to the specific edge type
        if edge_type in edge_dict:
            existing_src, existing_dst = edge_dict[edge_type]
            all_src = torch.cat([existing_src, torch.tensor(src_local, dtype=torch.long)])
            all_dst = torch.cat([existing_dst, torch.tensor(dst_local, dtype=torch.long)])
        else:
            all_src = torch.tensor(src_local, dtype=torch.long)
            all_dst = torch.tensor(dst_local, dtype=torch.long)
        
        edge_dict[edge_type] = (all_src, all_dst)
        
        # Create new graph
        num_nodes_dict = {ntype: self.graph.num_nodes(ntype) for ntype in self.graph.ntypes}
        new_graph = dgl.heterograph(edge_dict, num_nodes_dict=num_nodes_dict)
        
        # Copy all node features
        for ntype in self.graph.ntypes:
            for feature_name, feature_tensor in self.graph.nodes[ntype].data.items():
                new_graph.nodes[ntype].data[feature_name] = feature_tensor
        
        # Copy and extend edge features
        for etype in self.graph.canonical_etypes:
            current_edge_count = self.graph.num_edges(etype) if etype in self.graph.canonical_etypes else 0
            
            for feature_name, feature_tensor in self.graph.edges[etype].data.items():
                if etype == edge_type:
                    # This is the edge type we're adding to
                    edge_metadata_cols = [col for col in edges_df.columns if col not in ['src', 'dst']]
                    
                    if feature_name in edge_metadata_cols:
                        # Extend with new values
                        valid_edges_df = edges_df.iloc[valid_edges]
                        new_values = valid_edges_df[feature_name].values
                        
                        if valid_edges_df[feature_name].dtype in ['int64', 'float64', 'int32', 'float32']:
                            new_tensor = torch.cat([
                                feature_tensor,
                                torch.tensor(new_values, dtype=torch.float32)
                            ])
                        else:
                            new_tensor = torch.cat([
                                feature_tensor,
                                torch.zeros(len(valid_edges), dtype=feature_tensor.dtype)
                            ])
                    else:
                        # Feature not in new data, extend with defaults
                        new_tensor = torch.cat([
                            feature_tensor,
                            torch.zeros(len(valid_edges), dtype=feature_tensor.dtype)
                        ])
                    
                    new_graph.edges[etype].data[feature_name] = new_tensor
                else:
                    # Other edge types, just copy
                    new_graph.edges[etype].data[feature_name] = feature_tensor
        
        # Add new edge features for the target edge type
        edge_metadata_cols = [col for col in edges_df.columns if col not in ['src', 'dst']]
        for feature_name in edge_metadata_cols:
            if (edge_type not in self.graph.canonical_etypes or 
                feature_name not in self.graph.edges[edge_type].data):
                
                valid_edges_df = edges_df.iloc[valid_edges]
                if valid_edges_df[feature_name].dtype in ['int64', 'float64', 'int32', 'float32']:
                    # Create new tensor feature
                    existing_count = self.graph.num_edges(edge_type) if edge_type in self.graph.canonical_etypes else 0
                    feature_tensor = torch.zeros(existing_count + len(valid_edges), dtype=torch.float32)
                    feature_tensor[existing_count:] = torch.tensor(
                        valid_edges_df[feature_name].values, dtype=torch.float32)
                    new_graph.edges[edge_type].data[feature_name] = feature_tensor
        
        self.graph = new_graph
        
        return {
            'edges_added': len(valid_edges),
            'total_edges': new_graph.num_edges(),
            'edge_type': str(edge_type)
        }
    
    def bulk_delete_edges(self, edges_to_delete: Union[pd.DataFrame, List[Tuple[int, int]]], 
                     edge_type: Tuple[str, str, str] = None) -> Dict[str, Any]:
        """
        Efficiently delete specific edges in bulk from the graph.
        
        Args:
            edges_to_delete: DataFrame with ['src', 'dst'] columns (global IDs) or 
                            List of (src_global_id, dst_global_id) tuples
            edge_type: For heterogeneous graphs, specify (src_type, edge_rel, dst_type)
            
        Returns:
            Dict with deletion statistics
        """
        if self.graph is None:
            raise ValueError("No graph loaded")
        
        if edges_to_delete is None or len(edges_to_delete) == 0:
            return {'edges_deleted': 0, 'remaining_edges': self.graph.num_edges()}
        
        print(f"Bulk deleting edges...")
        start_time = time.time()
        
        if not self.is_heterogeneous():
            result = self._bulk_delete_edges_homogeneous(edges_to_delete)
        else:
            if edge_type is None:
                raise ValueError("edge_type is required for heterogeneous graphs")
            result = self._bulk_delete_edges_heterogeneous(edges_to_delete, edge_type)
        
        deletion_time = time.time() - start_time
        result['deletion_time'] = deletion_time
        
        print(f"Edge deletion completed in {deletion_time:.2f}s")
        print(f"Deleted {result['edges_deleted']} edges")
        
        return result

    def _bulk_delete_edges_homogeneous(self, edges_to_delete: Union[pd.DataFrame, List[Tuple[int, int]]]) -> Dict[str, Any]:
        """Handle bulk edge deletion for homogeneous graphs."""
        # Get current edges
        current_src, current_dst = self.graph.edges()
        current_src_list = current_src.tolist()
        current_dst_list = current_dst.tolist()
        total_edges = len(current_src_list)
        
        if total_edges == 0:
            return {'edges_deleted': 0, 'remaining_edges': 0}
        
        # Convert input to set of edge pairs
        if isinstance(edges_to_delete, pd.DataFrame):
            if 'src' not in edges_to_delete.columns or 'dst' not in edges_to_delete.columns:
                raise ValueError("DataFrame must contain 'src' and 'dst' columns")
            edges_to_delete_set = set(zip(edges_to_delete['src'], edges_to_delete['dst']))
        else:
            edges_to_delete_set = set(edges_to_delete)
        
        if not edges_to_delete_set:
            return {'edges_deleted': 0, 'remaining_edges': total_edges}
        
        # Find edge indices to keep
        edges_to_keep = []
        original_edge_indices_to_keep = []
        
        for i, (src, dst) in enumerate(zip(current_src_list, current_dst_list)):
            if (src, dst) not in edges_to_delete_set:
                edges_to_keep.append((src, dst))
                original_edge_indices_to_keep.append(i)
        
        # Create new graph with remaining edges
        if edges_to_keep:
            new_src, new_dst = zip(*edges_to_keep)
            new_src_tensor = torch.tensor(new_src, dtype=torch.long)
            new_dst_tensor = torch.tensor(new_dst, dtype=torch.long)
        else:
            new_src_tensor = torch.tensor([], dtype=torch.long)
            new_dst_tensor = torch.tensor([], dtype=torch.long)
        
        new_graph = dgl.graph((new_src_tensor, new_dst_tensor), num_nodes=self.graph.num_nodes())
        
        # Copy node features
        for feature_name, feature_tensor in self.graph.ndata.items():
            new_graph.ndata[feature_name] = feature_tensor
        
        # Copy edge features for remaining edges
        if original_edge_indices_to_keep:
            edge_indices_tensor = torch.tensor(original_edge_indices_to_keep, dtype=torch.long)
            for feature_name, feature_tensor in self.graph.edata.items():
                new_graph.edata[feature_name] = feature_tensor[edge_indices_tensor]
            
            # Update non-tensor edge data
            for feature_name, feature_data in self.edge_data.items():
                if isinstance(feature_data, np.ndarray):
                    self.edge_data[feature_name] = feature_data[original_edge_indices_to_keep]
                elif isinstance(feature_data, list):
                    self.edge_data[feature_name] = [feature_data[i] for i in original_edge_indices_to_keep]
        else:
            # No edges remaining, clear edge data
            self.edge_data = {}
        
        self.graph = new_graph
        edges_deleted = total_edges - len(edges_to_keep)
        
        return {
            'edges_deleted': edges_deleted,
            'remaining_edges': len(edges_to_keep),
            'total_original_edges': total_edges
        }

    def _bulk_delete_edges_heterogeneous(self, edges_to_delete: Union[pd.DataFrame, List[Tuple[int, int]]], 
                                    edge_type: Tuple[str, str, str]) -> Dict[str, Any]:
        """Handle bulk edge deletion for heterogeneous graphs."""
        src_type, edge_rel, dst_type = edge_type
        
        if edge_type not in self.graph.canonical_etypes:
            return {'edges_deleted': 0, 'remaining_edges': 0, 'error': f'Edge type {edge_type} not found'}
        
        # Get current edges for this edge type
        current_src, current_dst = self.graph.edges(etype=edge_type)
        current_src_list = current_src.tolist()
        current_dst_list = current_dst.tolist()
        total_edges = len(current_src_list)
        
        if total_edges == 0:
            return {'edges_deleted': 0, 'remaining_edges': 0}
        
        # Convert input to set of edge pairs and convert global to local IDs
        if isinstance(edges_to_delete, pd.DataFrame):
            edge_pairs = list(zip(edges_to_delete['src'], edges_to_delete['dst']))
        else:
            edge_pairs = edges_to_delete
        
        # Convert global IDs to local IDs for comparison
        local_edges_to_delete_set = set()
        for src_global, dst_global in edge_pairs:
            # Convert global IDs to local IDs
            if (src_global in self.global_to_local_mapping and 
                dst_global in self.global_to_local_mapping):
                
                src_node_type, src_local = self.global_to_local_mapping[src_global]
                dst_node_type, dst_local = self.global_to_local_mapping[dst_global]
                
                if src_node_type == src_type and dst_node_type == dst_type:
                    local_edges_to_delete_set.add((src_local, dst_local))
                else:
                    print(f"Warning: Edge ({src_global}, {dst_global}) type mismatch - expected ({src_type}, {dst_type}), got ({src_node_type}, {dst_node_type})")
            else:
                print(f"Warning: Edge ({src_global}, {dst_global}) contains non-existent nodes")
        
        if not local_edges_to_delete_set:
            return {'edges_deleted': 0, 'remaining_edges': total_edges}
        
        # Find edge indices to keep
        edges_to_keep = []
        original_edge_indices_to_keep = []
        
        for i, (src_local, dst_local) in enumerate(zip(current_src_list, current_dst_list)):
            if (src_local, dst_local) not in local_edges_to_delete_set:
                edges_to_keep.append((src_local, dst_local))
                original_edge_indices_to_keep.append(i)
        
        # Rebuild the graph with remaining edges
        edge_dict = {}
        
        # Copy all other edge types unchanged
        for etype in self.graph.canonical_etypes:
            if etype != edge_type:
                src, dst = self.graph.edges(etype=etype)
                edge_dict[etype] = (src, dst)
        
        # Add the modified edge type
        if edges_to_keep:
            new_src, new_dst = zip(*edges_to_keep)
            edge_dict[edge_type] = (torch.tensor(new_src, dtype=torch.long),
                                torch.tensor(new_dst, dtype=torch.long))
        # If no edges to keep for this type, don't add it to edge_dict
        
        # Create new graph
        num_nodes_dict = {ntype: self.graph.num_nodes(ntype) for ntype in self.graph.ntypes}
        new_graph = dgl.heterograph(edge_dict, num_nodes_dict=num_nodes_dict)
        
        # Copy all node features
        for ntype in self.graph.ntypes:
            for feature_name, feature_tensor in self.graph.nodes[ntype].data.items():
                new_graph.nodes[ntype].data[feature_name] = feature_tensor
        
        # Copy edge features
        for etype in new_graph.canonical_etypes:
            if etype == edge_type:
                # Copy features for remaining edges of the modified type
                if original_edge_indices_to_keep:
                    edge_indices_tensor = torch.tensor(original_edge_indices_to_keep, dtype=torch.long)
                    for feature_name, feature_tensor in self.graph.edges[etype].data.items():
                        new_graph.edges[etype].data[feature_name] = feature_tensor[edge_indices_tensor]
            else:
                # Copy features for unchanged edge types
                for feature_name, feature_tensor in self.graph.edges[etype].data.items():
                    new_graph.edges[etype].data[feature_name] = feature_tensor
        
        self.graph = new_graph
        edges_deleted = total_edges - len(edges_to_keep)
        
        return {
            'edges_deleted': edges_deleted,
            'remaining_edges': len(edges_to_keep),
            'total_original_edges': total_edges,
            'edge_type': str(edge_type)
        }
    
    # =============================================================================
    # 3. EFFICIENT K-HOP NEIGHBORHOOD RETRIEVAL
    # =============================================================================
    
    def get_k_hop_neighbors(self, seed_nodes: List[int], k: int, 
                          edge_types: List[Tuple[str, str, str]] = None,
                          bidirectional: bool = True) -> Dict[int, Set[int]]:
        """
        Get k-hop neighbors efficiently using BFS.
        
        Args:
            seed_nodes: Starting node IDs (global IDs for heterogeneous)
            k: Number of hops
            edge_types: For heterogeneous graphs, which edge types to follow
            bidirectional: Whether to consider both directions
            
        Returns:
            Dict mapping hop -> set of node IDs reached at that hop
        """
        if self.graph is None:
            raise ValueError("No graph loaded")
        
        if not self.is_heterogeneous():
            return self._get_k_hop_homogeneous(seed_nodes, k, bidirectional)
        else:
            return self._get_k_hop_heterogeneous(seed_nodes, k, edge_types, bidirectional)
    
    def _get_k_hop_homogeneous(self, seed_nodes: List[int], k: int, bidirectional: bool) -> Dict[int, Set[int]]:
        """K-hop neighbors for homogeneous graphs."""
        visited = set(seed_nodes)
        result = {0: set(seed_nodes)}
        current_frontier = set(seed_nodes)
        
        for hop in range(1, k + 1):
            next_frontier = set()
            
            for node in current_frontier:
                # Get neighbors efficiently
                neighbors = set()
                
                # Out-neighbors
                out_neighbors = self.graph.successors(node).tolist()
                neighbors.update(out_neighbors)
                
                # In-neighbors (if bidirectional)
                if bidirectional:
                    in_neighbors = self.graph.predecessors(node).tolist()
                    neighbors.update(in_neighbors)
                
                # Add unvisited neighbors
                new_neighbors = neighbors - visited
                next_frontier.update(new_neighbors)
            
            if not next_frontier:
                break
            
            result[hop] = next_frontier
            visited.update(next_frontier)
            current_frontier = next_frontier
        
        return result
    
    def _get_k_hop_heterogeneous(self, seed_nodes: List[int], k: int, 
                             edge_types: List[Tuple[str, str, str]] = None, 
                             bidirectional: bool = True) -> Dict[int, Set[int]]:
        """
        Get k-hop neighbors for heterogeneous graphs using global node IDs.
        
        Args:
            seed_nodes: List of global node IDs
            k: Number of hops
            edge_types: List of (src_type, edge_type, dst_type) to follow. If None, use all.
            bidirectional: Whether to follow edges in both directions
            
        Returns:
            Dict mapping hop_number -> set of global node IDs reached at that hop
        """
        if self.graph is None:
            raise ValueError("No graph loaded")
        
        if edge_types is None:
            edge_types = self.graph.canonical_etypes
        
        # Result dictionary
        result = {0: set(seed_nodes)}
        visited_global = set(seed_nodes)
        
        # Convert seed nodes to local IDs organized by node type
        current_frontier_by_type = defaultdict(set)
        
        for global_id in seed_nodes:
            if global_id in self.global_to_local_mapping:
                node_type, local_id = self.global_to_local_mapping[global_id]
                current_frontier_by_type[node_type].add(local_id)
            else:
                print(f"Warning: Global ID {global_id} not found in mapping")
        
        # Iterate through hops
        for hop in range(1, k + 1):
            next_frontier_by_type = defaultdict(set)
            
            # For each edge type
            for src_type, edge_rel, dst_type in edge_types:
                # Check if this edge type exists in the graph
                if (src_type, edge_rel, dst_type) not in self.graph.canonical_etypes:
                    continue
                
                # Forward direction: src_type -> dst_type
                if src_type in current_frontier_by_type:
                    neighbors = self._get_neighbors_efficient(
                        src_type, edge_rel, dst_type, 
                        current_frontier_by_type[src_type], 
                        direction='forward'
                    )
                    next_frontier_by_type[dst_type].update(neighbors)
                
                # Backward direction: dst_type -> src_type (if bidirectional)
                if bidirectional and dst_type in current_frontier_by_type:
                    neighbors = self._get_neighbors_efficient(
                        src_type, edge_rel, dst_type, 
                        current_frontier_by_type[dst_type], 
                        direction='backward'
                    )
                    next_frontier_by_type[src_type].update(neighbors)
            
            # Convert local IDs back to global IDs
            global_next = set()
            for node_type, local_ids in next_frontier_by_type.items():
                for local_id in local_ids:
                    try:
                        global_id = self.reverse_node_mapping[(node_type, local_id)]
                        if global_id not in visited_global:
                            global_next.add(global_id)
                    except ValueError:
                        # Handle case where local_id doesn't map back to global_id
                        continue
            
            # Break if no new nodes found
            if not global_next:
                break
            
            # Update results and visited set
            result[hop] = global_next
            visited_global.update(global_next)
            current_frontier_by_type = next_frontier_by_type
        
        return result
    
    def _get_neighbors_efficient(self, src_type: str, edge_rel: str, dst_type: str, 
                           source_nodes: Set[int], direction: str = 'forward') -> Set[int]:
        """
        Efficiently get neighbors for a specific edge type.
        
        Args:
            src_type: Source node type
            edge_rel: Edge relationship type
            dst_type: Destination node type
            source_nodes: Set of local node IDs to start from
            direction: 'forward' or 'backward'
            
        Returns:
            Set of local node IDs of neighbors
        """
        if not source_nodes:
            return set()
        
        edge_type_key = (src_type, edge_rel, dst_type)
        
        try:
            # Get all edges for this edge type
            src_edges, dst_edges = self.graph.edges(etype=edge_type_key)
            
            neighbors = set()
            
            if direction == 'forward':
                # Find all destination nodes connected to our source nodes
                for src_local in source_nodes:
                    mask = src_edges == src_local
                    dst_neighbors = dst_edges[mask].unique().tolist()
                    neighbors.update(dst_neighbors)
            
            elif direction == 'backward':
                # Find all source nodes connected to our destination nodes
                for dst_local in source_nodes:
                    mask = dst_edges == dst_local
                    src_neighbors = src_edges[mask].unique().tolist()
                    neighbors.update(src_neighbors)
            
            return neighbors
        
        except Exception as e:
            print(f"Error getting neighbors for {edge_type_key}: {e}")
            return set()
    
    # =============================================================================
    # 4. EFFICIENT SUBGRAPH EXTRACTION
    # =============================================================================
    
    def extract_subgraph(self, seed_nodes: List[int], k: int, 
                        edge_types: List[Tuple[str, str, str]] = None,
                        include_features: bool = True) -> dgl.DGLGraph:
        """
        Extract k-hop subgraph around seed nodes with proper node mapping.
        
        Args:
            seed_nodes: Starting node IDs
            k: Number of hops to include
            edge_types: For heterogeneous graphs
            include_features: Whether to copy node/edge features
            
        Returns:
            DGL subgraph with proper node ID mapping
        """
        # Get all nodes within k hops
        k_hop_nodes = self.get_k_hop_neighbors(seed_nodes, k, edge_types)
        all_nodes = set().union(*k_hop_nodes.values())
        
        if not self.is_heterogeneous():
            return self._extract_subgraph_homogeneous(all_nodes, include_features)
        else:
            return self._extract_subgraph_heterogeneous(all_nodes, include_features)
    
    def _extract_subgraph_homogeneous(self, node_set: Set[int], include_features: bool) -> dgl.DGLGraph:
        """Extract homogeneous subgraph."""
        node_list = sorted(list(node_set))
        node_tensor = torch.tensor(node_list, dtype=torch.long)
        
        subgraph = dgl.node_subgraph(self.graph, node_tensor)
        
        # The subgraph automatically includes features if include_features is True
        if not include_features:
            # Remove features
            subgraph.ndata.clear()
            subgraph.edata.clear()
        
        return subgraph

    def _extract_subgraph_heterogeneous(self, seed_nodes: List[int], k: int, 
                      edge_types: List[Tuple[str, str, str]] = None,
                      bidirectional: bool = True):
        """
        Extract k-hop heterogeneous subgraph around seed nodes using global IDs.
        
        Args:
            seed_nodes: List of global node IDs
            k: Number of hops
            edge_types: Edge types to follow
            bidirectional: Whether to follow edges in both directions
            
        Returns:
            DGL subgraph containing all nodes within k hops
        """
        # Get all nodes within k hops
        k_hop_result = self._get_k_hop_heterogeneous(seed_nodes, k, edge_types, bidirectional)
        
        # Collect all nodes (from all hops)
        all_global_nodes = set()
        for hop_nodes in k_hop_result.values():
            all_global_nodes.update(hop_nodes)
        
        # Convert to local IDs organized by node type
        nodes_by_type = defaultdict(list)
        for global_id in all_global_nodes:
            if global_id in self.global_to_local_mapping:
                node_type, local_id = self.global_to_local_mapping[global_id]
                nodes_by_type[node_type].append(local_id)
        
        # Create subgraph
        if len(self.graph.ntypes) == 1:
            # Homogeneous case
            node_tensor = torch.tensor(list(all_global_nodes), dtype=torch.long)
            subgraph = dgl.node_subgraph(self.graph, node_tensor)
        else:
            # Heterogeneous case
            node_dict = {}
            for node_type, local_ids in nodes_by_type.items():
                if local_ids:  # Only include types that have nodes
                    node_dict[node_type] = torch.tensor(local_ids, dtype=torch.long)
            
            if node_dict:
                subgraph = dgl.node_subgraph(self.graph, node_dict)
            else:
                # Return empty subgraph if no valid nodes
                subgraph = None
        
        return subgraph, k_hop_result
    
    # =============================================================================
    # 4.1 EFFICIENT SUBGRAPH MERGE #TODO: test more and check
    # =============================================================================
    
    def merge_subgraphs_optimal(self, subgraphs: List[dgl.DGLGraph], 
                           method: str = 'node_based') -> Tuple[dgl.DGLGraph, Dict[str, Any]]:
        """
        Merge multiple subgraphs into a single subgraph optimally using DGL functions.
        
        Args:
            subgraphs: List of DGL subgraphs to merge
            method: 'union' (fast, for compatible graphs) or 'node_based' (robust, any graphs)
            
        Returns:
            Tuple of (merged_subgraph, merge_statistics)
        """
        if not subgraphs:
            raise ValueError("No subgraphs provided")
        
        if len(subgraphs) == 1:
            return subgraphs[0], {'method': 'single', 'num_input_graphs': 1}
        
        print(f"Merging {len(subgraphs)} subgraphs using method '{method}'...")
        start_time = time.time()
        
        if method == 'union':
            merged_graph, stats = self._merge_with_union(subgraphs)
        elif method == 'node_based':
            merged_graph, stats = self._merge_with_node_extraction(subgraphs)
        else:
            raise ValueError(f"Unknown method: {method}")
        
        merge_time = time.time() - start_time
        stats['merge_time'] = merge_time
        stats['method'] = method
        
        print(f"Merge completed in {merge_time:.2f}s")
        print(f"Result: {merged_graph.num_nodes()} nodes, {merged_graph.num_edges()} edges")
        
        return merged_graph, stats

    def _merge_with_union(self, subgraphs: List[dgl.DGLGraph]) -> Tuple[dgl.DGLGraph, Dict[str, Any]]:
        """
        Fast merge using DGL's union function (works best with compatible subgraphs).
        """
        try:
            # Use DGL's efficient union operation
            merged_graph = dgl.union(subgraphs)
            
            stats = {
                'num_input_graphs': len(subgraphs),
                'input_nodes': [g.num_nodes() for g in subgraphs],
                'input_edges': [g.num_edges() for g in subgraphs],
                'total_input_nodes': sum(g.num_nodes() for g in subgraphs),
                'total_input_edges': sum(g.num_edges() for g in subgraphs),
                'final_nodes': merged_graph.num_nodes(),
                'final_edges': merged_graph.num_edges(),
                'deduplication_ratio': 1.0 - (merged_graph.num_nodes() / sum(g.num_nodes() for g in subgraphs))
            }
            
            return merged_graph, stats
            
        except Exception as e:
            print(f"Union method failed: {e}")
            print("Falling back to node-based method...")
            return self._merge_with_node_extraction(subgraphs)

    def _merge_with_node_extraction(self, subgraphs: List[dgl.DGLGraph]) -> Tuple[dgl.DGLGraph, Dict[str, Any]]:
        """
        Robust merge by collecting all nodes and re-extracting from main graph.
        """
        # Collect all unique nodes from all subgraphs
        if isinstance(subgraphs[0], dgl.DGLHeteroGraph) and len(subgraphs[0].ntypes) > 1:
            return self._merge_heterogeneous_node_based(subgraphs)
        else:
            return self._merge_homogeneous_node_based(subgraphs)

    def _merge_homogeneous_node_based(self, subgraphs: List[dgl.DGLGraph]) -> Tuple[dgl.DGLGraph, Dict[str, Any]]:
        """Merge homogeneous subgraphs by collecting nodes."""
        all_nodes = set()
        
        # Collect all node IDs
        for subgraph in subgraphs:
            # Get original node IDs from subgraph
            if dgl.NID in subgraph.ndata:
                original_ids = subgraph.ndata[dgl.NID].tolist()
                all_nodes.update(original_ids)
            else:
                # If no NID, assume consecutive numbering
                all_nodes.update(range(subgraph.num_nodes()))
        
        # Extract merged subgraph from main graph
        node_tensor = torch.tensor(list(all_nodes), dtype=torch.long)
        merged_graph = dgl.node_subgraph(self.graph, node_tensor)
        
        stats = {
            'num_input_graphs': len(subgraphs),
            'unique_nodes_collected': len(all_nodes),
            'final_nodes': merged_graph.num_nodes(),
            'final_edges': merged_graph.num_edges(),
        }
        
        return merged_graph, stats

    def _merge_heterogeneous_node_based(self, subgraphs: List[dgl.DGLGraph]) -> Tuple[dgl.DGLGraph, Dict[str, Any]]:
        """Merge heterogeneous subgraphs by collecting nodes per type."""
        nodes_by_type = defaultdict(set)
        
        # Collect all node IDs organized by type
        for subgraph in subgraphs:
            for ntype in subgraph.ntypes:
                if subgraph.num_nodes(ntype) > 0:
                    if dgl.NID in subgraph.nodes[ntype].data:
                        original_ids = subgraph.nodes[ntype].data[dgl.NID].tolist()
                        nodes_by_type[ntype].update(original_ids)
                    else:
                        # If no NID, assume consecutive numbering
                        nodes_by_type[ntype].update(range(subgraph.num_nodes(ntype)))
        
        # Convert to tensor format for DGL
        node_dict = {}
        for ntype, node_set in nodes_by_type.items():
            if node_set:  # Only include types with nodes
                node_dict[ntype] = torch.tensor(list(node_set), dtype=torch.long)
        
        # Extract merged subgraph from main graph
        if node_dict:
            merged_graph = dgl.node_subgraph(self.graph, node_dict)
        else:
            # Return empty graph if no nodes
            merged_graph = dgl.heterograph({}, num_nodes_dict={})
        
        stats = {
            'num_input_graphs': len(subgraphs),
            'nodes_by_type': {ntype: len(nodes) for ntype, nodes in nodes_by_type.items()},
            'final_nodes': merged_graph.num_nodes(),
            'final_edges': merged_graph.num_edges(),
        }
        
        return merged_graph, stats

    def merge_k_hop_subgraphs(self, seed_groups: List[List[int]], k: int, 
                            edge_types: List[Tuple[str, str, str]] = None,
                            merge_method: str = 'node_based') -> Tuple[dgl.DGLGraph, Dict[str, Any]]:
        """
        Extract k-hop subgraphs around multiple seed groups and merge them.
        
        Args:
            seed_groups: List of seed node groups (each group is a list of global IDs)
            k: Number of hops for each subgraph
            edge_types: Edge types to follow
            merge_method: Method for merging ('union' or 'node_based')
            
        Returns:
            Tuple of (merged_subgraph, detailed_statistics)
        """
        print(f"Extracting and merging {len(seed_groups)} k-hop subgraphs (k={k})...")
        
        # Extract individual subgraphs
        individual_subgraphs = []
        extraction_stats = []
        
        for i, seed_group in enumerate(seed_groups):
            print(f"  Extracting subgraph {i+1}/{len(seed_groups)} (seeds: {len(seed_group)})...")
            
            subgraph, k_hop_result = self.extract_subgraph(seed_group, k, edge_types)
            
            if subgraph is not None:
                individual_subgraphs.append(subgraph)
                extraction_stats.append({
                    'seed_group_size': len(seed_group),
                    'k_hop_result': k_hop_result,
                    'subgraph_nodes': subgraph.num_nodes(),
                    'subgraph_edges': subgraph.num_edges()
                })
            else:
                print(f"    Warning: No subgraph extracted for group {i+1}")
        
        if not individual_subgraphs:
            raise ValueError("No valid subgraphs extracted")
        
        # Merge all subgraphs
        merged_subgraph, merge_stats = self.merge_subgraphs_optimal(individual_subgraphs, merge_method)
        
        # Combine statistics
        detailed_stats = {
            'extraction_stats': extraction_stats,
            'merge_stats': merge_stats,
            'summary': {
                'num_seed_groups': len(seed_groups),
                'k_hops': k,
                'individual_subgraphs': len(individual_subgraphs),
                'final_nodes': merged_subgraph.num_nodes(),
                'final_edges': merged_subgraph.num_edges(),
                'node_sharing_detected': merge_stats.get('deduplication_ratio', 0) > 0
            }
        }
        
        return merged_subgraph, detailed_stats

    def smart_subgraph_merger(self, node_groups: List[List[int]], 
                         connection_strategy: str = 'k_hop',
                         k: int = 2,
                         min_shared_nodes: int = 1) -> Tuple[dgl.DGLGraph, Dict[str, Any]]:
        """
        Intelligently merge subgraphs with different connection strategies.
        
        Args:
            node_groups: Groups of global node IDs to connect
            connection_strategy: 'k_hop', 'shortest_path', or 'direct_neighbors'
            k: Parameter for k-hop strategy
            min_shared_nodes: Minimum shared nodes required for merging
            
        Returns:
            Tuple of (merged_subgraph, connection_analysis)
        """
        print(f"Smart merging of {len(node_groups)} node groups using '{connection_strategy}' strategy...")
        
        if connection_strategy == 'k_hop':
            return self.merge_k_hop_subgraphs(node_groups, k, merge_method='node_based')
        
        elif connection_strategy == 'shortest_path':
            return self._merge_with_shortest_paths(node_groups)
        
        elif connection_strategy == 'direct_neighbors':
            return self._merge_with_direct_neighbors(node_groups)
        
        else:
            raise ValueError(f"Unknown connection strategy: {connection_strategy}")

    def _merge_with_shortest_paths(self, node_groups: List[List[int]]) -> Tuple[dgl.DGLGraph, Dict[str, Any]]:
        """Merge by finding shortest paths between groups."""
        all_path_nodes = set()
        path_stats = []
        
        # Add all seed nodes
        for group in node_groups:
            all_path_nodes.update(group)
        
        # Find shortest paths between groups
        for i in range(len(node_groups)):
            for j in range(i + 1, len(node_groups)):
                # Find path between representative nodes from each group
                src_node = node_groups[i][0]  # First node from group i
                dst_node = node_groups[j][0]  # First node from group j
                
                path = self.find_shortest_path_heterogeneous(src_node, dst_node, max_hops=5)
                
                if path:
                    all_path_nodes.update(path)
                    path_stats.append({
                        'group_i': i,
                        'group_j': j,
                        'path_length': len(path),
                        'path_nodes': path
                    })
        
        # Convert to subgraph
        if isinstance(self.graph, dgl.DGLHeteroGraph) and len(self.graph.ntypes) > 1:
            # Heterogeneous case: organize by node type
            nodes_by_type = defaultdict(list)
            for global_id in all_path_nodes:
                if global_id in self.global_to_local_mapping:
                    node_type, local_id = self.global_to_local_mapping[global_id]
                    nodes_by_type[node_type].append(local_id)
            
            node_dict = {ntype: torch.tensor(local_ids, dtype=torch.long) 
                        for ntype, local_ids in nodes_by_type.items() if local_ids}
            merged_subgraph = dgl.node_subgraph(self.graph, node_dict)
        else:
            # Homogeneous case
            node_tensor = torch.tensor(list(all_path_nodes), dtype=torch.long)
            merged_subgraph = dgl.node_subgraph(self.graph, node_tensor)
        
        stats = {
            'total_path_nodes': len(all_path_nodes),
            'paths_found': len(path_stats),
            'path_details': path_stats,
            'final_nodes': merged_subgraph.num_nodes(),
            'final_edges': merged_subgraph.num_edges()
        }
        
        return merged_subgraph, stats

    def _merge_with_direct_neighbors(self, node_groups: List[List[int]]) -> Tuple[dgl.DGLGraph, Dict[str, Any]]:
        """Merge by including direct neighbors of all seed nodes."""
        all_nodes = set()
        
        # Add all seed nodes
        for group in node_groups:
            all_nodes.update(group)
        
        # Add direct neighbors of all seed nodes
        neighbor_stats = []
        for group in node_groups:
            group_neighbors = set()
            for node_id in group:
                neighbors_result = self.get_k_hop_heterogeneous([node_id], k=1)
                if 1 in neighbors_result:
                    group_neighbors.update(neighbors_result[1])
            
            all_nodes.update(group_neighbors)
            neighbor_stats.append({
                'group_size': len(group),
                'neighbors_added': len(group_neighbors)
            })
        
        # Extract subgraph
        if isinstance(self.graph, dgl.DGLHeteroGraph) and len(self.graph.ntypes) > 1:
            nodes_by_type = defaultdict(list)
            for global_id in all_nodes:
                if global_id in self.global_to_local_mapping:
                    node_type, local_id = self.global_to_local_mapping[global_id]
                    nodes_by_type[node_type].append(local_id)
            
            node_dict = {ntype: torch.tensor(local_ids, dtype=torch.long) 
                        for ntype, local_ids in nodes_by_type.items() if local_ids}
            merged_subgraph = dgl.node_subgraph(self.graph, node_dict)
        else:
            node_tensor = torch.tensor(list(all_nodes), dtype=torch.long)
            merged_subgraph = dgl.node_subgraph(self.graph, node_tensor)
        
        stats = {
            'neighbor_stats': neighbor_stats,
            'total_nodes_with_neighbors': len(all_nodes),
            'final_nodes': merged_subgraph.num_nodes(),
            'final_edges': merged_subgraph.num_edges()
        }
        
        return merged_subgraph, stats
    
    # =============================================================================
    # 5. META-PATH QUERIES (FIXED AND OPTIMIZED) #TODO: test more and check
    # =============================================================================
    
    def find_meta_paths(self, start_nodes: List[int], meta_path: List[str], 
                       max_paths_per_node: int = 100, max_length: int = 10) -> Dict[int, List[List[int]]]:
        """
        Find meta-paths efficiently using BFS instead of DFS.
        
        Args:
            start_nodes: Starting node IDs (global)
            meta_path: List of edge types defining the path pattern
            max_paths_per_node: Maximum paths to return per starting node
            max_length: Maximum path length to prevent infinite loops
            
        Returns:
            Dict mapping start_node -> list of paths (each path is a list of global node IDs)
        """
        if not self.is_heterogeneous():
            raise ValueError("Meta-path queries require heterogeneous graphs")
        
        if len(meta_path) == 0 or len(meta_path) > max_length:
            return {node: [] for node in start_nodes}
        
        result = {}
        
        for start_node in start_nodes:
            paths = self._find_paths_from_node(start_node, meta_path, max_paths_per_node)
            result[start_node] = paths
        
        return result

    def _find_paths_from_node(self, start_node: int, meta_path: List[str], max_paths: int) -> List[List[int]]:
        """Find meta-paths from a single start node using BFS with global IDs."""

        start_type, start_local = self.global_to_local_mapping[start_node]
        
        # BFS to find paths
        queue = deque([(start_type, start_local, [start_node], 0)])  # (current_type, local_id, path, step)
        paths = []
        
        while queue and len(paths) < max_paths:
            current_type, current_local, current_path, step = queue.popleft()
            
            if step >= len(meta_path):
                if len(current_path) > 1:  # Only add paths with at least 2 nodes
                    paths.append(current_path)
                continue
            
            # Find the appropriate canonical edge type
            edge_rel = meta_path[step]
            matching_etypes = [
                (src, etype, dst) for src, etype, dst in self.graph.canonical_etypes
                if etype == edge_rel and src == current_type
            ]
            
            for src_type, etype, dst_type in matching_etypes:
                if self.graph.num_edges((src_type, etype, dst_type)) == 0:
                    continue
                
                try:
                    # Get neighbors efficiently using DGL's edge access
                    edges = self.graph.edges(etype=(src_type, etype, dst_type))
                    src_edges, dst_edges = edges
                    
                    # Find neighbors of current node
                    mask = src_edges == current_local
                    neighbors = dst_edges[mask].unique()
                    
                    for neighbor_local in neighbors.tolist():
                        # Convert to global ID
                        neighbor_global = self.reverse_node_mapping.get((dst_type, neighbor_local))
                        if neighbor_global is not None and neighbor_global not in current_path:
                            new_path = current_path + [neighbor_global]
                            queue.append((dst_type, neighbor_local, new_path, step + 1))
                            
                            if len(paths) >= max_paths:
                                break
                    
                except Exception as e:
                    print(f"Error processing edge type {etype}: {e}")
                    continue
                
                if len(paths) >= max_paths:
                    break
            
            if len(paths) >= max_paths:
                break
        
        return paths
    
    def get_meta_path_neighbors(self, start_nodes: List[int], meta_path: List[str], 
                               return_paths: bool = False) -> Dict[int, List[int]]:
        """
        Get all neighbors reachable via a specific meta-path.
        
        Args:
            start_nodes: Starting node global IDs
            meta_path: List of edge types defining the path pattern
            return_paths: If True, return the full paths instead of just end nodes
            
        Returns:
            Dict mapping start_node -> list of reachable nodes (or paths if return_paths=True)
        """
        all_paths = self.find_meta_paths(start_nodes, meta_path, max_paths_per_node=1000)
        
        result = {}
        for start_node, paths in all_paths.items():
            if return_paths:
                result[start_node] = paths
            else:
                # Return only the end nodes
                end_nodes = list(set(path[-1] for path in paths if len(path) > 1))
                result[start_node] = end_nodes
        
        return result

    def find_shortest_path_heterogeneous(self, start_global_id: int, end_global_id: int, 
                                   max_hops: int = 5, 
                                   edge_types: List[Tuple[str, str, str]] = None) -> List[int]:
        """
        Find shortest path between two nodes using global IDs.
        
        Args:
            start_global_id: Starting node global ID
            end_global_id: Ending node global ID
            max_hops: Maximum number of hops to search
            edge_types: Edge types to follow
            
        Returns:
            List of global node IDs representing the shortest path (empty if no path found)
        """
        if start_global_id == end_global_id:
            return [start_global_id]
        
        # BFS to find shortest path
        from collections import deque
        
        queue = deque([(start_global_id, [start_global_id])])
        visited = {start_global_id}
        
        for hop in range(max_hops):
            if not queue:
                break
            
            # Process all nodes at current level
            next_level_queue = deque()
            
            while queue:
                current_global_id, path = queue.popleft()
                
                # Get 1-hop neighbors
                neighbors_result = self.get_k_hop_heterogeneous([current_global_id], 1, edge_types)
                
                if 1 in neighbors_result:
                    for neighbor_global_id in neighbors_result[1]:
                        if neighbor_global_id == end_global_id:
                            return path + [neighbor_global_id]
                        
                        if neighbor_global_id not in visited:
                            visited.add(neighbor_global_id)
                            next_level_queue.append((neighbor_global_id, path + [neighbor_global_id]))
            
            queue = next_level_queue
        
        return []  # No path found
    
    def validate_meta_path(self, meta_path: List[str]) -> bool:
        """
        Validate if a meta-path is possible in the current graph schema.
        
        Args:
            meta_path: List of edge types
            
        Returns:
            True if the meta-path is valid, False otherwise
        """
        if not self.is_heterogeneous():
            return False
        
        if not meta_path:
            return True
        
        # Check if consecutive edge types can be connected
        available_edges = defaultdict(set)  # edge_type -> set of (src_type, dst_type)
        
        for src_type, edge_type, dst_type in self.graph.canonical_etypes:
            available_edges[edge_type].add((src_type, dst_type))
        
        # Try to find a valid path through the schema
        possible_current_types = set(self.graph.ntypes)
        
        for edge_type in meta_path:
            if edge_type not in available_edges:
                return False
            
            next_possible_types = set()
            for src_type, dst_type in available_edges[edge_type]:
                if src_type in possible_current_types:
                    next_possible_types.add(dst_type)
            
            if not next_possible_types:
                return False
            
            possible_current_types = next_possible_types
        
        return True
    
    def find_paths_between_nodes(self, source_nodes: List[int], target_nodes: List[int], 
                                meta_path: List[str], max_paths: int = 100) -> Dict[Tuple[int, int], List[List[int]]]:
        """
        Find meta-paths between specific source and target nodes.
        
        Args:
            source_nodes: List of source node global IDs
            target_nodes: List of target node global IDs
            meta_path: List of edge types defining the path pattern
            max_paths: Maximum paths to return per source-target pair
            
        Returns:
            Dict mapping (source_global_id, target_global_id) -> list of paths
        """
        if not self.is_heterogeneous():
            raise ValueError("Meta-path queries require heterogeneous graphs")
        
        target_set = set(target_nodes)
        result = {}
        
        for source_node in source_nodes:
            all_paths = self._find_paths_from_node(source_node, meta_path, max_paths * len(target_nodes))
            
            for target_node in target_nodes:
                # Filter paths that end at the target node
                matching_paths = [path for path in all_paths if path[-1] == target_node][:max_paths]
                if matching_paths:
                    result[(source_node, target_node)] = matching_paths
    
    # =============================================================================
    # 6. EFFICIENT QUERYING
    # =============================================================================
    
    def query_nodes_by_feature(
        self,
        entity_type: str | None,
        feature_name: str,
        condition: str,
        value: Union[float, int, List],
        return_features: Optional[List[str]] = None
    ) -> Dict[str, torch.Tensor]:
        """
        Fast vectorized node querying by feature conditions.
        
        Args:
            feature_name: Name of the feature to query on
            condition: One of 'eq', 'gt', 'lt', 'gte', 'lte', 'in', 'between'
            value: Value(s) to compare against
            return_features: List of features to return for matching nodes
        
        Returns:
            Dict with 'node_ids' and requested features
        """
        if feature_name not in self.graph.ndata:
            raise ValueError(f"Feature {feature_name} not found in node data")
        
        feature_tensor = self.graph.ndata[feature_name][entity_type]
        
        # Vectorized condition evaluation
        if condition == 'eq':
            mask = feature_tensor == value
        elif condition == 'gt':
            mask = feature_tensor > value
        elif condition == 'lt':
            mask = feature_tensor < value
        elif condition == 'gte':
            mask = feature_tensor >= value
        elif condition == 'lte':
            mask = feature_tensor <= value
        elif condition == 'in':
            value_tensor = torch.tensor(value, dtype=feature_tensor.dtype)
            mask = torch.isin(feature_tensor, value_tensor)
        elif condition == 'between':
            if len(value) != 2:
                raise ValueError("'between' condition requires list of 2 values")
            mask = (feature_tensor >= value[0]) & (feature_tensor <= value[1])
        else:
            raise ValueError(f"Unsupported condition: {condition}")
        
        # Get matching node IDs
        matching_nodes = torch.where(mask)[0]
        
        result = {'node_ids': matching_nodes}
        
        # Return requested features efficiently
        if return_features:
            for feat in return_features:
                if feat in self.graph.ndata.keys():
                    result[feat] = self.graph.ndata[feat][entity_type][matching_nodes]
                else:
                    result[feat] = self.node_data[entity_type].get(feat, torch.tensor([]))[matching_nodes]
        
        return result
    
    def query_nodes_multi_condition(
        self,
        conditions: List[Dict],
        logic: str = 'and',
        return_features: Optional[List[str]] = None
    ) -> Dict[str, torch.Tensor]:
        """
        Query nodes with multiple conditions efficiently.
        
        Args:
            conditions: List of condition dicts with keys: 'feature', 'condition', 'value'
            logic: 'and' or 'or' for combining conditions
            return_features: Features to return for matching nodes
        
        Example:
            conditions = [
                {'feature': 'age', 'condition': 'gt', 'value': 25},
                {'feature': 'score', 'condition': 'gte', 'value': 0.8}
            ]
        """
        if not conditions:
            return {'node_ids': torch.arange(self.graph.num_nodes())}
        
        # Evaluate all conditions
        masks = []
        for cond in conditions:
            single_result = self.query_nodes_by_feature(
                cond['feature'], cond['condition'], cond['value']
            )
            # Convert node IDs to boolean mask
            mask = torch.zeros(self.graph.num_nodes(), dtype=torch.bool)
            mask[single_result['node_ids']] = True  
            masks.append(mask)
        
        # Combine masks
        if logic == 'and':
            final_mask = masks[0]
            for mask in masks[1:]:
                final_mask = final_mask & mask
        elif logic == 'or':
            final_mask = masks[0]
            for mask in masks[1:]:
                final_mask = final_mask | mask
        else:
            raise ValueError("Logic must be 'and' or 'or'")
        
        matching_nodes = torch.where(final_mask)[0]
        result = {'node_ids': matching_nodes}
        
        # Return requested features
        if return_features:
            for feat in return_features:
                if feat in self.graph.ndata:
                    result[feat] = self.graph.ndata[feat][matching_nodes]
        
        return result
    
    def get_top_nodes_by_feature(
        self,
        entity_type: str | None,
        feature_name: str,
        top_k: int = 10,
        ascending: bool = False,
        return_features: Optional[List[str]] = None
    ) -> Dict[str, torch.Tensor]:
        """
        Efficiently get top-k nodes by a feature value.
        """
        if feature_name not in self.graph.ndata:
            raise ValueError(f"Feature {feature_name} not found")
        
        feature_tensor = self.graph.ndata[feature_name][entity_type]
        
        # Use PyTorch's efficient topk function
        values, indices = torch.topk(
            feature_tensor, 
            k=min(top_k, len(feature_tensor)),
            largest=not ascending
        )
        
        result = {
            'node_ids': indices,
            feature_name: values
        }
        
        # Add other requested features
        if return_features:
            for feat in return_features:
                if feat in self.graph.ndata.keys() and feat != feature_name:
                    result[feat] = self.graph.ndata[feat][entity_type][indices]
                else:
                    result[feat] = self.node_data[entity_type].get(feat, torch.tensor([]))[indices]
        
        return result

    def query_edges(self, edge_type: Tuple[str, str, str] = None, 
                   feature_filters: Dict[str, Any] = None, limit: int = None) -> pd.DataFrame:
        """Query edges with optional filters."""
        if self.graph is None:
            raise ValueError("No graph loaded")
        
        if not self.is_heterogeneous():
            return self._query_edges_homogeneous(feature_filters, limit)
        else:
            return self._query_edges_heterogeneous(edge_type, feature_filters, limit)
    
    def _query_edges_homogeneous(self, feature_filters: Dict[str, Any], limit: int) -> pd.DataFrame:
        """Query homogeneous graph edges."""
        edges = self.graph.edges()
        src, dst = edges
        
        data = {
            'src': src.numpy(),
            'dst': dst.numpy(),
            'edge_id': list(range(len(src)))
        }
        
        # Add edge features
        for feature_name in self.graph.edata.keys():
            data[feature_name] = self.graph.edata[feature_name].numpy()
        
        df = pd.DataFrame(data)
        
        # Apply filters
        if feature_filters:
            for feature, value in feature_filters.items():
                if feature in df.columns:
                    df = df[df[feature] == value]
        
        if limit:
            df = df.head(limit)
        
        return df
    
    def _query_edges_heterogeneous(self, edge_type: Tuple[str, str, str], 
                                 feature_filters: Dict[str, Any], limit: int) -> pd.DataFrame:
        """Query heterogeneous graph edges."""
        if edge_type is None:
            edge_type = self.graph.canonical_etypes[0]
        
        edges = self.graph.edges(etype=edge_type)
        src, dst = edges
        
        # Convert local IDs to global IDs
        src_type, etype, dst_type = edge_type
        src_global = []
        dst_global = []
        
        for s, d in zip(src.tolist(), dst.tolist()):
            src_gid = self.reverse_node_mapping.get((src_type, s))
            dst_gid = self.reverse_node_mapping.get((dst_type, d))
            if src_gid is not None and dst_gid is not None:
                src_global.append(src_gid)
                dst_global.append(dst_gid)
        
        data = {
            'src': src_global,
            'dst': dst_global,
            'src_type': [src_type] * len(src_global),
            'edge_type': [etype] * len(src_global),
            'dst_type': [dst_type] * len(src_global)
        }
        
        # Add edge features
        if edge_type in self.graph.canonical_etypes:
            for feature_name in self.graph.edges[edge_type].data.keys():
                feature_data = self.graph.edges[edge_type].data[feature_name]
                if len(feature_data) == len(src_global):
                    data[feature_name] = feature_data.numpy()
        
        df = pd.DataFrame(data)
        
        # Apply filters
        if feature_filters:
            for feature, value in feature_filters.items():
                if feature in df.columns:
                    df = df[df[feature] == value]
        
        if limit:
            df = df.head(limit)
        
        return df
    
    def get_degree_statistics(self, force_refresh: bool = False) -> Dict[str, torch.Tensor]:
        """
        Cached degree statistics computation.
        """
        if self._degree_cache is None or force_refresh:
            in_degrees = self.graph.in_degrees()
            out_degrees = self.graph.out_degrees()
            total_degrees = in_degrees + out_degrees
            
            self._degree_cache = {
                'in_degrees': in_degrees,
                'out_degrees': out_degrees,
                'total_degrees': total_degrees
            }
        
        return self._degree_cache
    
    def query_nodes_by_degree(
        self,
        degree_type: str = 'total',  # 'in', 'out', 'total'
        condition: str = 'gt',
        value: int = 0,
        return_features: Optional[List[str]] = None
    ) -> Dict[str, torch.Tensor]:
        """
        Query nodes by their degree efficiently.
        """
        degree_stats = self.get_degree_statistics()
        
        if degree_type == 'in':
            degrees = degree_stats['in_degrees']
        elif degree_type == 'out':
            degrees = degree_stats['out_degrees']
        elif degree_type == 'total':
            degrees = degree_stats['total_degrees']
        else:
            raise ValueError("degree_type must be 'in', 'out', or 'total'")
        
        # Apply condition
        if condition == 'eq':
            mask = degrees == value
        elif condition == 'gt':
            mask = degrees > value
        elif condition == 'lt':
            mask = degrees < value
        elif condition == 'gte':
            mask = degrees >= value
        elif condition == 'lte':
            mask = degrees <= value
        else:
            raise ValueError(f"Unsupported condition: {condition}")
        
        matching_nodes = torch.where(mask)[0]
        result = {
            'node_ids': matching_nodes,
            f'{degree_type}_degree': degrees[matching_nodes]
        }
        
        if return_features:
            for feat in return_features:
                if feat in self.graph.ndata:
                    result[feat] = self.graph.ndata[feat][matching_nodes]
        
        return result
    
    def find_common_neighbors(
        self,
        node1: int,
        node2: int,
        return_features: Optional[List[str]] = None
    ) -> Dict[str, torch.Tensor]:
        """
        Find common neighbors of two nodes efficiently.
        """
        neighbors1 = set(self.graph.successors(node1).tolist() + 
                        self.graph.predecessors(node1).tolist())
        neighbors2 = set(self.graph.successors(node2).tolist() + 
                        self.graph.predecessors(node2).tolist())
        
        common = neighbors1.intersection(neighbors2)
        common_tensor = torch.tensor(list(common), dtype=torch.long)
        
        result = {'node_ids': common_tensor}
        
        if return_features:
            for feat in return_features:
                if feat in self.graph.ndata:
                    result[feat] = self.graph.ndata[feat][common_tensor]
        
        return result
    
    # =============================================================================
    # 7. UTILITY METHODS
    # =============================================================================
    
    def is_heterogeneous(self) -> bool:
        """Check if the graph is heterogeneous."""
        return isinstance(self.graph, dgl.DGLHeteroGraph) and len(self.graph.ntypes) > 1
    
    def save_graph(self, filepath: str):
        """Save the DGL graph and metadata efficiently."""
        dgl.save_graphs(filepath, [self.graph])
        
        metadata = {
            'node_data': self.node_data,
            'edge_data': self.edge_data,
            'global_to_local_mapping': self.global_to_local_mapping,
            'reverse_node_mapping': self.reverse_node_mapping
        }
        
        with open(filepath.replace('.bin', '_metadata.pkl'), 'wb') as f:
            pickle.dump(metadata, f)
        
        print(f"Graph saved to {filepath}")
    
    def load_graph(self, filepath: str):
        """Load the DGL graph and metadata efficiently."""
        graphs, _ = dgl.load_graphs(filepath)
        self.graph = graphs[0]
        
        try:
            with open(filepath.replace('.bin', '_metadata.pkl'), 'rb') as f:
                metadata = pickle.load(f)
                self.node_data = metadata.get('node_data', {})
                self.edge_data = metadata.get('edge_data', {})
                self.global_to_local_mapping = metadata.get('global_to_local_mapping', {})
                self.reverse_node_mapping = metadata.get('reverse_node_mapping', {})
        except FileNotFoundError:
            print("No metadata file found, only graph structure loaded")
        
        print(f"Graph loaded from {filepath}")
        return self.graph
    
    def get_graph_statistics(self) -> Dict[str, Any]:
        """Get comprehensive graph statistics."""
        if self.graph is None:
            return {}
        
        stats = {}
        
        if self.is_heterogeneous():
            stats['type'] = 'heterogeneous'
            stats['node_types'] = {ntype: self.graph.num_nodes(ntype) for ntype in self.graph.ntypes}
            stats['edge_types'] = {str(etype): self.graph.num_edges(etype) 
                                 for etype in self.graph.canonical_etypes}
            stats['total_nodes'] = sum(stats['node_types'].values())
            stats['total_edges'] = sum(stats['edge_types'].values())
            
            # Degree statistics per node type
            stats['degree_stats'] = {}
            for ntype in self.graph.ntypes:
                # Calculate degrees for each node type across all relevant edge types
                total_in_degrees = torch.zeros(self.graph.num_nodes(ntype))
                total_out_degrees = torch.zeros(self.graph.num_nodes(ntype))
                
                # Sum degrees across all edge types involving this node type
                for src_type, etype, dst_type in self.graph.canonical_etypes:
                    try:
                        if dst_type == ntype:  # This node type is destination
                            in_deg = self.graph.in_degrees(etype=(src_type, etype, dst_type))
                            if len(in_deg) == self.graph.num_nodes(ntype):
                                total_in_degrees += in_deg.float()
                        
                        if src_type == ntype:  # This node type is source
                            out_deg = self.graph.out_degrees(etype=(src_type, etype, dst_type))
                            if len(out_deg) == self.graph.num_nodes(ntype):
                                total_out_degrees += out_deg.float()
                    except Exception:
                        # Skip if there's an issue with this edge type
                        continue
                
                stats['degree_stats'][ntype] = {
                    'avg_in_degree': float(total_in_degrees.mean()) if len(total_in_degrees) > 0 else 0,
                    'avg_out_degree': float(total_out_degrees.mean()) if len(total_out_degrees) > 0 else 0,
                    'max_in_degree': int(total_in_degrees.max()) if len(total_in_degrees) > 0 else 0,
                    'max_out_degree': int(total_out_degrees.max()) if len(total_out_degrees) > 0 else 0,
                }
            
            # Edge type specific statistics
            stats['edge_type_stats'] = {}
            for etype in self.graph.canonical_etypes:
                src_type, edge_rel, dst_type = etype
                num_edges = self.graph.num_edges(etype)
                stats['edge_type_stats'][str(etype)] = {
                    'num_edges': num_edges,
                    'density': float(num_edges / (self.graph.num_nodes(src_type) * self.graph.num_nodes(dst_type))) 
                               if self.graph.num_nodes(src_type) > 0 and self.graph.num_nodes(dst_type) > 0 else 0
                }
        else:
            stats['type'] = 'homogeneous'
            stats['total_nodes'] = self.graph.num_nodes()
            stats['total_edges'] = self.graph.num_edges()
            
            # Degree statistics
            in_degrees = self.graph.in_degrees().float()
            out_degrees = self.graph.out_degrees().float()
            
            stats['avg_in_degree'] = float(in_degrees.mean())
            stats['avg_out_degree'] = float(out_degrees.mean())
            stats['max_in_degree'] = int(in_degrees.max())
            stats['max_out_degree'] = int(out_degrees.max())
            stats['density'] = float(2 * stats['total_edges'] / (stats['total_nodes'] * (stats['total_nodes'] - 1))) if stats['total_nodes'] > 1 else 0
        
        return stats
    
    def get_node_features(self, node_ids: List[int], feature_names: List[str] = None, 
                         node_type: str = None) -> Dict[str, np.ndarray]:
        """Retrieve node features efficiently."""
        if not self.is_heterogeneous():
            return self._get_node_features_homogeneous(node_ids, feature_names)
        else:
            return self._get_node_features_heterogeneous(node_ids, feature_names, node_type)
    
    def _get_node_features_homogeneous(self, node_ids: List[int], 
                                     feature_names: List[str]) -> Dict[str, np.ndarray]:
        """Get features for homogeneous graph."""
        if feature_names is None:
            feature_names = list(self.graph.ndata.keys())
        
        result = {}
        node_tensor = torch.tensor(node_ids, dtype=torch.long)
        
        for feature in feature_names:
            if feature in self.graph.ndata:
                result[feature] = self.graph.ndata[feature][node_tensor].numpy()
        
        return result
    
    def _get_node_features_heterogeneous(self, node_ids: List[int], feature_names: List[str], 
                                       node_type: str) -> Dict[str, np.ndarray]:
        """Get features for heterogeneous graph."""
        if node_type is None:
            # Group by node type
            result_by_type = {}
            for ntype in self.graph.ntypes:
                type_nodes = [nid for nid in node_ids if self.global_to_local_mapping[nid][0] == ntype]
                if type_nodes:
                    result_by_type[ntype] = self._get_node_features_heterogeneous(type_nodes, feature_names, ntype)
            return result_by_type
        
        # Get features for specific node type
        type_node_ids = [nid for nid in node_ids if self.global_to_local_mapping[nid][0] == node_type]
        
        if feature_names is None:
            feature_names = list(self.graph.nodes[node_type].data.keys())
        
        result = {}
        
        # Convert to local IDs
        local_ids = []
        for global_id in type_node_ids:
            for (nt, local_id), gid in self.reverse_node_mapping.items():
                if gid == global_id and nt == node_type:
                    local_ids.append(local_id)
                    break
        
        if local_ids:
            local_tensor = torch.tensor(local_ids, dtype=torch.long)
            for feature in feature_names:
                if feature in self.graph.nodes[node_type].data:
                    result[feature] = self.graph.nodes[node_type].data[feature][local_tensor].numpy()
        
        return result