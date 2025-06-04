import torch
import pandas as pd
from typing import Dict, List, Tuple, Any, Optional, Union
import logging

logger = logging.getLogger(__name__)

class QueryMixin:
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
    