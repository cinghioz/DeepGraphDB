import dgl
from typing import Dict, Any, Optional, Tuple
from dataloader import CoverageBasedDataLoader

class GraphDatabaseBase:
    """Base class containing core attributes and basic functionality"""
    
    def __init__(self):
        self.graph = None
        self.node_data = {}
        self.edge_data = {}
        self.reverse_node_mapping = {}  # Maps (node_type, local_id) -> global_id
        self.global_to_local_mapping = {} # Maps global_id -> (node_type, local_id)
        self.node_types_mapping = {} # string -> int
        self.edge_types_mapping = {} # string -> int
        self._degree_cache: Optional[Dict[str, Any]] = None
        self.node_features = {}
    
    def is_heterogeneous(self) -> bool:
        """Check if the graph is heterogeneous."""
        return isinstance(self.graph, dgl.DGLHeteroGraph) and len(self.graph.ntypes) > 1
    
    def validate_graph_loaded(self):
        """Validate that a graph has been loaded."""
        if self.graph is None:
            raise ValueError("No graph loaded. Please load a graph first.")
        
    def set_mappings(self, node_types: Dict[str, int], edge_types: Dict[str, int]):
        self.node_types_mapping = node_types
        self.edge_types_mapping = edge_types

    def set_global_to_local_mapping(self, mapping: Dict[Tuple[str, int], int]):
        self.global_to_local_mapping = mapping
        self.reverse_node_mapping = dict(zip(mapping.values(), mapping.keys()))
    