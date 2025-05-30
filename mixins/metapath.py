import dgl
import torch
from typing import Dict, List, Tuple
from collections import defaultdict, deque
import logging

logger = logging.getLogger(__name__)

class MetaPathMixin:
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
    