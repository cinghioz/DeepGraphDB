import dgl
import torch
import pandas as pd
import numpy as np
from typing import Dict, List, Set, Tuple, Any, Union
from collections import defaultdict
import time
import logging

logger = logging.getLogger(__name__)

class CrudMixin:
    """Mixin for CRUD operations on DGL graphs"""
    
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
        
        # # Generate new global IDs (consecutive from current max)
        # if len(self.global_to_local_mapping) > 0:
        #     next_global_id = max(self.global_to_local_mapping.keys()) + 1
        # else:
        #     next_global_id = 0
        
        # new_global_ids = list(range(next_global_id, next_global_id + num_new_nodes))
        # new_local_ids = list(range(current_num_nodes, new_total_nodes))

        if len(self.global_to_local_mapping) > 0:
            existing_global_ids = set(self.global_to_local_mapping.keys())
            max_global_id = max(existing_global_ids)
            
            # Find holes in the range [0, max_global_id]
            all_possible_ids = set(range(max_global_id + 1))
            holes = sorted(list(all_possible_ids - existing_global_ids))
            
            # Use holes first, then continue with new IDs
            new_global_ids = []
            
            # Fill holes first
            holes_to_use = min(len(holes), num_new_nodes)
            new_global_ids.extend(holes[:holes_to_use])
            
            # If we need more IDs after filling holes
            remaining_nodes = num_new_nodes - holes_to_use
            if remaining_nodes > 0:
                next_global_id = max_global_id + 1
                new_global_ids.extend(range(next_global_id, next_global_id + remaining_nodes))
        else:
            # No existing nodes, start from 0
            new_global_ids = list(range(num_new_nodes))

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
        
        # Generate new global IDs by filling holes first
        if len(self.global_to_local_mapping) > 0:
            existing_global_ids = set(self.global_to_local_mapping.keys())
            max_global_id = max(existing_global_ids)
            
            # Find holes in the range [0, max_global_id]
            all_possible_ids = set(range(max_global_id + 1))
            holes = sorted(list(all_possible_ids - existing_global_ids))
            
            # Use holes first, then continue with new IDs
            new_global_ids = []
            
            # Fill holes first
            holes_to_use = min(len(holes), num_new_nodes)
            new_global_ids.extend(holes[:holes_to_use])
            
            # If we need more IDs after filling holes
            remaining_nodes = num_new_nodes - holes_to_use
            if remaining_nodes > 0:
                next_global_id = max_global_id + 1
                new_global_ids.extend(range(next_global_id, next_global_id + remaining_nodes))
        else:
            # No existing nodes, start from 0
            new_global_ids = list(range(num_new_nodes))

        new_local_ids = list(range(current_num_nodes, new_total_nodes))

        # # Generate new global IDs
        # if len(self.global_to_local_mapping) > 0:
        #     next_global_id = max(self.global_to_local_mapping.keys()) + 1
        # else:
        #     next_global_id = 0
        
        # new_global_ids = list(range(next_global_id, next_global_id + num_new_nodes))
        # new_local_ids = list(range(current_num_nodes, new_total_nodes))
        
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
        # self._recompute_id_mappings()
        
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
    