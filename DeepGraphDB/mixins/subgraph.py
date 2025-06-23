import dgl
import torch
from typing import Dict, List, Set, Tuple, Any, Optional
from collections import defaultdict
import time
import logging

logger = logging.getLogger(__name__)

class SubgraphMixin:
    def extract_subgraph(self, seed_nodes: List[int], k: int, 
                        edge_types: List[Tuple[str, str, str]] = None,
                        include_features: bool = True, max_neighbors_per_hop: Optional[List[int]] = None) -> dgl.DGLGraph:
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
        k_hop_nodes = self.get_k_hop_neighbors(seed_nodes, k, edge_types, max_neighbors=max_neighbors_per_hop)
        
        if not self.is_heterogeneous():
            return self._extract_subgraph_homogeneous(k_hop_nodes, include_features)
        else:
            return self._extract_subgraph_heterogeneous(k_hop_nodes, include_features)
    
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

    def _extract_subgraph_heterogeneous(self, k_hop_result: Dict[int, List[int]], include_features: bool):
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
        # k_hop_result = self._get_k_hop_heterogeneous(seed_nodes, k, edge_types, bidirectional, max_neighbors_per_hop=max_neighbors_per_hop)
        
        # Collect all nodes (from all hops)
        all_global_nodes = set()
        for hop_nodes in k_hop_result.values():
            all_global_nodes.update(hop_nodes)
        
        # Convert to local IDs organized by node type
        nodes_by_type = defaultdict(list)
        gids_by_type = defaultdict(list)
        
        for global_id in all_global_nodes:
            if global_id in self.global_to_local_mapping:
                node_type, local_id = self.global_to_local_mapping[global_id]
                nodes_by_type[node_type].append(local_id)
                gids_by_type[node_type].append(global_id)
        
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
        
        return subgraph, k_hop_result, gids_by_type
    
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
    