import dgl
import torch
import pandas as pd
import numpy as np
from typing import Dict, List, Set, Tuple
from collections import defaultdict
import logging

logger = logging.getLogger(__name__)

class NeighborhoodMixin:
    """Mixin class for neighborhood operations on graphs."""
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
    