import dgl
import torch
import pandas as pd
import numpy as np
from typing import Dict, List, Set, Tuple
from collections import defaultdict
import logging

logger = logging.getLogger(__name__)

import random
from collections import defaultdict
from typing import List, Dict, Set, Tuple, Optional, Union


#TODO: il numero di vicino va semplato prima nomn dopo avere tutto il vicinato per ottimizzare!
import random
from collections import defaultdict
from typing import List, Dict, Set, Tuple, Optional, Union

class NeighborhoodMixin:
    """Mixin class for neighborhood operations on graphs."""
    def get_k_hop_neighbors(self, seed_nodes: List[int], k: int, 
                          edge_types: List[Tuple[str, str, str]] = None,
                          bidirectional: bool = True,
                          max_neighbors: Optional[Union[int, List[int]]] = None,
                          random_seed: Optional[int] = None) -> Dict[int, Set[int]]:
        """
        Get k-hop neighbors efficiently using BFS with optional random sampling per node.
        
        Args:
            seed_nodes: Starting node IDs (global IDs for heterogeneous)
            k: Number of hops
            edge_types: For heterogeneous graphs, which edge types to follow
            bidirectional: Whether to consider both directions
            max_neighbors: Maximum neighbors to sample per node per hop. Can be:
                         - int: same limit for all hops
                         - List[int]: specific limit for each hop (length should be >= k)
                         - None: no sampling (default behavior)
            random_seed: Random seed for reproducible sampling
            
        Returns:
            Dict mapping hop -> set of node IDs reached at that hop
        """
        if self.graph is None:
            raise ValueError("No graph loaded")
        
        # Set random seed if provided
        if random_seed is not None:
            random.seed(random_seed)
        
        # Process max_neighbors parameter
        max_neighbors_per_hop = self._process_max_neighbors_param(max_neighbors, k)
        
        if not self.is_heterogeneous():
            return self._get_k_hop_homogeneous(seed_nodes, k, bidirectional, max_neighbors_per_hop)
        else:
            return self._get_k_hop_heterogeneous(seed_nodes, k, edge_types, bidirectional, max_neighbors_per_hop)
    
    def _process_max_neighbors_param(self, max_neighbors: Optional[Union[int, List[int]]], k: int) -> Optional[List[int]]:
        """
        Process the max_neighbors parameter into a list format.
        
        Args:
            max_neighbors: The input parameter
            k: Number of hops
            
        Returns:
            List of max neighbors per hop, or None if no sampling
        """
        if max_neighbors is None:
            return None
        
        if isinstance(max_neighbors, int):
            return [max_neighbors] * k
        
        if isinstance(max_neighbors, list):
            if len(max_neighbors) < k:
                raise ValueError(f"max_neighbors list length ({len(max_neighbors)}) must be >= k ({k})")
            return max_neighbors[:k]  # Take only first k elements
        
        raise ValueError("max_neighbors must be int, list, or None")
    
    def _sample_neighbors_for_node(self, neighbors: Set[int], max_count: Optional[int]) -> Set[int]:
        """
        Randomly sample neighbors for a single node if max_count is specified.
        
        Args:
            neighbors: Set of all neighbors for a single node
            max_count: Maximum number to sample, or None for no sampling
            
        Returns:
            Sampled set of neighbors
        """
        if max_count is None or len(neighbors) <= max_count:
            return neighbors
        
        return set(random.sample(list(neighbors), max_count))
    
    def _get_k_hop_homogeneous(self, seed_nodes: List[int], k: int, bidirectional: bool, 
                             max_neighbors_per_hop: Optional[List[int]]) -> Dict[int, Set[int]]:
        """K-hop neighbors for homogeneous graphs with per-node sampling."""
        visited = set(seed_nodes)
        result = {0: set(seed_nodes)}
        current_frontier = set(seed_nodes)
        
        for hop in range(1, k + 1):
            next_frontier = set()
            max_count = max_neighbors_per_hop[hop - 1] if max_neighbors_per_hop else None
            
            for node in current_frontier:
                # Get neighbors efficiently for this specific node
                neighbors = set()
                
                # Out-neighbors
                out_neighbors = self.graph.successors(node).tolist()
                neighbors.update(out_neighbors)
                
                # In-neighbors (if bidirectional)
                if bidirectional:
                    in_neighbors = self.graph.predecessors(node).tolist()
                    neighbors.update(in_neighbors)
                
                # Remove already visited neighbors
                new_neighbors = neighbors - visited
                
                # Sample neighbors for this specific node
                sampled_neighbors = self._sample_neighbors_for_node(new_neighbors, max_count)
                next_frontier.update(sampled_neighbors)
            
            if not next_frontier:
                break
            
            result[hop] = next_frontier
            visited.update(next_frontier)
            current_frontier = next_frontier
        
        return result
    
    def _get_k_hop_heterogeneous(self, seed_nodes: List[int], k: int, 
                             edge_types: List[Tuple[str, str, str]] = None, 
                             bidirectional: bool = True,
                             max_neighbors_per_hop: Optional[List[int]] = None) -> Dict[int, Set[int]]:
        """
        Get k-hop neighbors for heterogeneous graphs using global node IDs with per-node sampling.
        
        Args:
            seed_nodes: List of global node IDs
            k: Number of hops
            edge_types: List of (src_type, edge_type, dst_type) to follow. If None, use all.
            bidirectional: Whether to follow edges in both directions
            max_neighbors_per_hop: List of max neighbors to sample per node per hop
            
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
            max_count = max_neighbors_per_hop[hop - 1] if max_neighbors_per_hop else None
            
            # For each node type in current frontier
            for current_node_type, current_nodes in current_frontier_by_type.items():
                
                # Process each node individually
                for current_node_local in current_nodes:
                    node_neighbors_by_type = defaultdict(set)
                    
                    # For each edge type
                    for src_type, edge_rel, dst_type in edge_types:
                        # Check if this edge type exists in the graph
                        if (src_type, edge_rel, dst_type) not in self.graph.canonical_etypes:
                            continue
                        
                        # Forward direction: current_node_type -> dst_type
                        if current_node_type == src_type:
                            neighbors = self._get_neighbors_for_single_node(
                                src_type, edge_rel, dst_type, 
                                current_node_local, 
                                direction='forward'
                            )
                            node_neighbors_by_type[dst_type].update(neighbors)
                        
                        # Backward direction: current_node_type <- src_type (if bidirectional)
                        if bidirectional and current_node_type == dst_type:
                            neighbors = self._get_neighbors_for_single_node(
                                src_type, edge_rel, dst_type, 
                                current_node_local, 
                                direction='backward'
                            )
                            node_neighbors_by_type[src_type].update(neighbors)
                    
                    # Convert to global IDs and filter out visited nodes
                    all_node_neighbors_global = set()
                    for neighbor_type, neighbor_locals in node_neighbors_by_type.items():
                        for local_id in neighbor_locals:
                            try:
                                global_id = self.reverse_node_mapping[(neighbor_type, local_id)]
                                if global_id not in visited_global:
                                    all_node_neighbors_global.add(global_id)
                            except (KeyError, ValueError):
                                continue
                    
                    # Sample neighbors for this specific node
                    sampled_neighbors_global = self._sample_neighbors_for_node(
                        all_node_neighbors_global, max_count
                    )
                    
                    # Add sampled neighbors back to next frontier by type
                    for global_id in sampled_neighbors_global:
                        if global_id in self.global_to_local_mapping:
                            node_type, local_id = self.global_to_local_mapping[global_id]
                            next_frontier_by_type[node_type].add(local_id)
            
            # Convert all next frontier to global IDs for result
            global_next = set()
            for node_type, local_ids in next_frontier_by_type.items():
                for local_id in local_ids:
                    try:
                        global_id = self.reverse_node_mapping[(node_type, local_id)]
                        global_next.add(global_id)
                    except (KeyError, ValueError):
                        continue
            
            # Break if no new nodes found
            if not global_next:
                break
            
            # Update results and visited set
            result[hop] = global_next
            visited_global.update(global_next)
            current_frontier_by_type = next_frontier_by_type
        
        return result
    
    def _get_neighbors_for_single_node(self, src_type: str, edge_rel: str, dst_type: str, 
                                     source_node: int, direction: str = 'forward') -> Set[int]:
        """
        Efficiently get neighbors for a single node for a specific edge type.
        
        Args:
            src_type: Source node type
            edge_rel: Edge relationship type
            dst_type: Destination node type
            source_node: Single local node ID to start from
            direction: 'forward' or 'backward'
            
        Returns:
            Set of local node IDs of neighbors for this single node
        """
        edge_type_key = (src_type, edge_rel, dst_type)
        
        try:
            # Get all edges for this edge type
            src_edges, dst_edges = self.graph.edges(etype=edge_type_key)
            
            neighbors = set()
            
            if direction == 'forward':
                # Find all destination nodes connected to this source node
                mask = src_edges == source_node
                dst_neighbors = dst_edges[mask].unique().tolist()
                neighbors.update(dst_neighbors)
            
            elif direction == 'backward':
                # Find all source nodes connected to this destination node
                mask = dst_edges == source_node
                src_neighbors = src_edges[mask].unique().tolist()
                neighbors.update(src_neighbors)
            
            return neighbors
        
        except Exception as e:
            print(f"Error getting neighbors for {edge_type_key}: {e}")
            return set()
    
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