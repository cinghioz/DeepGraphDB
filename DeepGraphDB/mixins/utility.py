import dgl
import torch
import numpy as np
from typing import Dict, List, Any
import pickle
import logging

logger = logging.getLogger(__name__)

class UtilityMixin:
    def is_heterogeneous(self) -> bool:
        """Check if the graph is heterogeneous."""
        return isinstance(self.graph, dgl.DGLHeteroGraph) and len(self.graph.ntypes) > 1
    
    def save_graph(self, filepath: str):
        """Save the DGL graph and metadata efficiently."""
        dgl.save_graphs(filepath, [self.graph])
        
        metadata = {
            'node_data': self.node_data,
            'edge_data': self.edge_data,
            'node_types_mapping': self.node_types_mapping,
            'edge_types_mapping': self.edge_types_mapping,
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
                self.node_types_mapping = metadata.get('node_types_mapping', {})
                self.edge_types_mapping = metadata.get('edge_types_mapping', {})
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