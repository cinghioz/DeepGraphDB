# %%
import pandas as pd
import numpy as np
from typing import Dict, Tuple
from collections import defaultdict
import time

class PrimeKGLoader:
    """
    Prepares PrimeKG data for efficient loading into DGL heterogeneous graphs.
    Each node type gets sequential IDs starting from 0.
    """
    
    def __init__(self):
        self.node_type_mapping = {}  # string -> int
        self.relationship_type_mapping = {}  # string -> int
        self.reverse_node_type_mapping = {}  # int -> string
        self.reverse_relationship_type_mapping = {}  # int -> string
        self.global_to_local_mapping = {}  # For reference: global_id -> (node_type, local_id)
        
    def load_and_prepare_primekg(self, nodes_csv_path: str, edges_csv_path: str):
        """
        Load PrimeKG data and prepare it for bulk_load_heterogeneous_graph.
        Each node type gets sequential IDs starting from 0.
        
        Args:
            nodes_csv_path: Path to nodes CSV file
            edges_csv_path: Path to edges CSV file
            
        Returns:
            Tuple of (node_types_dict, edge_types_dict) ready for DGL loading
        """
        print("Loading PrimeKG data...")
        start_time = time.time()
        
        # Load raw data
        print("  Reading CSV files...")
        nodes_df = pd.read_csv(nodes_csv_path, low_memory=False)
        edges_df = pd.read_csv(edges_csv_path, low_memory=False)
        
        print(f"  Loaded {len(nodes_df):,} nodes and {len(edges_df):,} edges")
        
        # Create type mappings
        print("  Creating type mappings...")
        self._create_type_mappings(nodes_df, edges_df)
        
        # Prepare node data (sequential IDs starting from 0 for each type)
        print("  Preparing node data...")
        node_types_dict = self._prepare_node_data(nodes_df)
        
        # Prepare edge data (using local IDs)
        print("  Preparing edge data...")
        edge_types_dict = self._prepare_edge_data(edges_df, nodes_df)
        
        total_time = time.time() - start_time
        print(f"\nData preparation completed in {total_time:.2f}s")
        
        # Print summary
        self._print_summary(node_types_dict, edge_types_dict)
        
        return node_types_dict, edge_types_dict, self.global_to_local_mapping
    
    def _create_type_mappings(self, nodes_df: pd.DataFrame, edges_df: pd.DataFrame):
        """Create mappings between string types and integer representations."""
        
        # Node type mappings
        unique_node_types = sorted(nodes_df['node_type'].unique())
        self.node_type_mapping = {node_type: i for i, node_type in enumerate(unique_node_types)}
        self.reverse_node_type_mapping = {i: node_type for node_type, i in self.node_type_mapping.items()}
        
        # Relationship type mappings
        unique_rel_types = sorted(edges_df['relationship_type'].unique())
        self.relationship_type_mapping = {rel_type: i for i, rel_type in enumerate(unique_rel_types)}
        self.reverse_relationship_type_mapping = {i: rel_type for rel_type, i in self.relationship_type_mapping.items()}
        
        print(f"    Found {len(unique_node_types)} node types: {unique_node_types}")
        print(f"    Found {len(unique_rel_types)} relationship types: {unique_rel_types}")
    
    def _prepare_node_data(self, nodes_df: pd.DataFrame) -> Dict[str, pd.DataFrame]:
        """
        Group nodes by type and prepare DataFrames with sequential IDs starting from 0.
        
        Returns:
            Dict mapping node_type_string -> DataFrame with columns ['node_id', 'name', 'metadata_source', 'node_type_id', 'original_global_id']
        """
        node_types_dict = {}
        
        # Add numeric type ID to nodes
        nodes_df_copy = nodes_df.copy()
        nodes_df_copy['node_type_id'] = nodes_df_copy['node_type'].map(self.node_type_mapping)
        
        # Group by node type and assign sequential IDs starting from 0
        for node_type_str, group_df in nodes_df_copy.groupby('node_type'):
            # Sort by original ID for consistency
            group_df = group_df.sort_values('id').reset_index(drop=True)
            
            # Create sequential IDs starting from 0
            num_nodes = len(group_df)
            
            # Build global to local mapping for this node type
            global_ids = group_df['id'].values
            local_ids = np.arange(num_nodes)  # 0, 1, 2, ..., num_nodes-1
            
            # Store the mapping for edge processing
            for local_id, global_id in zip(local_ids, global_ids):
                self.global_to_local_mapping[global_id] = (node_type_str, local_id)
            
            # Prepare DataFrame for DGL
            prepared_df = pd.DataFrame({
                'node_id': local_ids,  # Sequential IDs starting from 0
                'name': group_df['name'].values,
                'metadata_source': group_df['metadata_source'].values,
                'node_type_id': group_df['node_type_id'].values,
                'original_global_id': global_ids  # Keep original for reference
            })
            
            node_types_dict[node_type_str] = prepared_df
            print(f"    {node_type_str}: {num_nodes:,} nodes (IDs: 0 to {num_nodes-1})")
            
        return node_types_dict
    
    def _prepare_edge_data(self, edges_df: pd.DataFrame, nodes_df: pd.DataFrame) -> Dict[Tuple[str, str, str], pd.DataFrame]:
        """
        Prepare edge data grouped by (src_type, edge_type, dst_type) using local IDs.
        
        Returns:
            Dict mapping (src_type, edge_type, dst_type) -> DataFrame with columns ['src', 'dst', 'relationship_type_id']
        """
        # Create node ID to type mapping for fast lookup
        node_id_to_type = dict(zip(nodes_df['id'], nodes_df['node_type']))
        
        # Add relationship type IDs
        edges_df_copy = edges_df.copy()
        edges_df_copy['relationship_type_id'] = edges_df_copy['relationship_type'].map(self.relationship_type_mapping)
        
        # Add source and target node types
        edges_df_copy['src_type'] = edges_df_copy['source_id'].map(node_id_to_type)
        edges_df_copy['dst_type'] = edges_df_copy['target_id'].map(node_id_to_type)
        
        # Filter out edges with unknown nodes
        valid_mask = (edges_df_copy['src_type'].notna()) & (edges_df_copy['dst_type'].notna())
        valid_edges = edges_df_copy[valid_mask]
        
        if len(valid_edges) < len(edges_df_copy):
            print(f"    Warning: Filtered out {len(edges_df_copy) - len(valid_edges)} edges with unknown nodes")
        
        # Group by (src_type, relationship_type, dst_type)
        edge_types_dict = {}
        
        for (src_type, rel_type, dst_type), group_df in valid_edges.groupby(['src_type', 'relationship_type', 'dst_type']):
            print(f"    Processing {src_type} --[{rel_type}]--> {dst_type}: {len(group_df):,} edges")
            
            # VECTORIZED APPROACH - Much faster than loops
            group_df_reset = group_df.reset_index(drop=True)
            
            # Create mapping functions for this specific edge type
            src_type_mapping = {global_id: local_id for global_id, (nt, local_id) in self.global_to_local_mapping.items() if nt == src_type}
            dst_type_mapping = {global_id: local_id for global_id, (nt, local_id) in self.global_to_local_mapping.items() if nt == dst_type}
            
            # Vectorized mapping using pandas map
            group_df_reset['src_local'] = group_df_reset['source_id'].map(src_type_mapping)
            group_df_reset['dst_local'] = group_df_reset['target_id'].map(dst_type_mapping)
            
            # Filter valid edges (both src and dst must be mapped)
            valid_mask = (group_df_reset['src_local'].notna()) & (group_df_reset['dst_local'].notna())
            valid_edges_df = group_df_reset[valid_mask]
            
            if len(valid_edges_df) == 0:
                print(f"      Warning: No valid edges found for {src_type}-{rel_type}->{dst_type}")
                continue
            
            # Create edge DataFrame with local node IDs
            edge_df = pd.DataFrame({
                'src': valid_edges_df['src_local'].astype(int).values,  # Local IDs (0-based for each node type)
                'dst': valid_edges_df['dst_local'].astype(int).values,  # Local IDs (0-based for each node type)
                'relationship_type_id': valid_edges_df['relationship_type_id'].values,
                'original_src_id': valid_edges_df['source_id'].values,  # Keep original for reference
                'original_dst_id': valid_edges_df['target_id'].values   # Keep original for reference
            })
            
            edge_types_dict[(src_type, rel_type, dst_type)] = edge_df
            print(f"      Created {len(edge_df):,} valid edges")
            
        return edge_types_dict
    
    def _print_summary(self, node_types_dict: Dict[str, pd.DataFrame], 
                      edge_types_dict: Dict[Tuple[str, str, str], pd.DataFrame]):
        """Print summary of prepared data."""
        print("\n" + "="*60)
        print("PRIMEKG DATA PREPARATION SUMMARY")
        print("="*60)
        
        print("\nNode Type Mappings:")
        for str_type, int_type in self.node_type_mapping.items():
            count = len(node_types_dict.get(str_type, []))
            print(f"  {int_type}: {str_type} ({count:,} nodes, IDs: 0 to {count-1})")
        
        print("\nRelationship Type Mappings:")
        for str_type, int_type in self.relationship_type_mapping.items():
            print(f"  {int_type}: {str_type}")
        
        print("\nPrepared Node Types:")
        total_nodes = 0
        for node_type, df in node_types_dict.items():
            min_id = df['node_id'].min()
            max_id = df['node_id'].max()
            print(f"  {node_type}: {len(df):,} nodes (local IDs: {min_id} to {max_id})")
            total_nodes += len(df)
        print(f"  TOTAL: {total_nodes:,} nodes")
        
        print("\nPrepared Edge Types:")
        total_edges = 0
        for (src_type, edge_type, dst_type), df in edge_types_dict.items():
            print(f"  {src_type} --[{edge_type}]--> {dst_type}: {len(df):,} edges")
            total_edges += len(df)
        print(f"  TOTAL: {total_edges:,} edges")
        
        print("\nData Format Verification:")
        for node_type, df in node_types_dict.items():
            assert df['node_id'].min() == 0, f"Node IDs for {node_type} don't start at 0!"
            assert df['node_id'].max() == len(df) - 1, f"Node IDs for {node_type} are not sequential!"
            print(f"  ✅ {node_type}: Sequential IDs 0 to {len(df)-1}")
        
        print("="*60)
    
    def get_type_mappings(self):
        """Return the type mappings for reference."""
        return {
            'node_types': self.node_type_mapping,
            'relationship_types': self.relationship_type_mapping,
            'reverse_node_types': self.reverse_node_type_mapping,
            'reverse_relationship_types': self.reverse_relationship_type_mapping
        }
    
    def get_global_to_local_mapping(self):
        """Return the global to local ID mapping for reference."""
        return self.global_to_local_mapping.copy()
    
    def global_id_to_local(self, global_id: int) -> Tuple[str, int]:
        """Convert a global node ID to (node_type, local_id)."""
        if global_id in self.global_to_local_mapping:
            return self.global_to_local_mapping[global_id]
        else:
            raise ValueError(f"Global ID {global_id} not found in mapping")
    
    def local_id_to_global(self, node_type: str, local_id: int) -> int:
        """Convert (node_type, local_id) to global node ID."""
        for global_id, (nt, lid) in self.global_to_local_mapping.items():
            if nt == node_type and lid == local_id:
                return global_id
        raise ValueError(f"Local ID ({node_type}, {local_id}) not found in mapping")

# %%
from DeepGraphDB import DeepGraphDB
import torch

db = DeepGraphDB()
# Initialize the loader
loader = PrimeKGLoader()

# Load and prepare data
nodes_csv = "data/nodes.csv"  # Replace with your actual path
edges_csv = "data/edges.csv"  # Replace with your actual path

node_types_dict, edge_types_dict, mapping = loader.load_and_prepare_primekg(nodes_csv, edges_csv)
    
# Get type mappings for reference
mappings = loader.get_type_mappings()
print("\nType mappings created:")
print("Node types:", mappings['node_types'])
print("Relationship types:", mappings['relationship_types'])

# Verify data format
print("\nData format verification:")
for node_type, df in node_types_dict.items():
    print(f"  {node_type}: node_id range {df['node_id'].min()}-{df['node_id'].max()}")

# Now you can use this data with your DGL graph analyzer
print("\nReady to load into DGL!")
print("Use: analyzer.bulk_load_heterogeneous_graph(node_types_dict, edge_types_dict)")
db.bulk_load_heterogeneous_graph(node_types_dict, edge_types_dict)
db.set_mappings(loader.node_type_mapping, loader.relationship_type_mapping)
db.set_global_to_local_mapping(mapping)

# x = torch.rand(max(db.global_to_local_mapping.keys())+1, 256)
x = torch.load("/home/cc/PHD/dglframework/DeepKG/start_feats.pt")
db.load_node_features_for_gnn(torch.tensor(x))

# %%
# db.save_graph("/home/cc/PHD/dglframework/DeepKG/DeepGraphDB/graphs/primekg.bin")

# %%
import torch
import dgl
import numpy as np
from collections import defaultdict
from sklearn.model_selection import train_test_split
from DeepGraphDB.gnns.heteroSAGEattn import AdvancedHeteroLinkPredictor, compute_loss

in_feats = {ntype: x.shape[1] for ntype in db.graph.ntypes}

# Choose multiple edge types for prediction
target_etypes = [ctype for ctype in db.graph.canonical_etypes if (ctype[0] == "geneprotein" or ctype[2] in "geneprotein") and db.graph.num_edges(ctype) > 3000]
# target_etypes = [ctype for ctype in db.graph.canonical_etypes if ctype[0] in target_entities and ctype[2] in target_entities]
# target_etypes = [('disease', 'contraindication', 'drug'), ('disease', 'indication', 'drug'), ('drug', 'contraindication', 'disease'), ('drug', 'indication', 'disease')]

print(f"Target edge types for prediction: {target_etypes}")

hidden_feats = 512
out_feats = 512

model = AdvancedHeteroLinkPredictor(
    node_types=db.graph.ntypes,  # All node types in the graph
    edge_types=db.graph.etypes,  # All edge types for GNN layers
    canonical_etypes=db.graph.canonical_etypes,  # All canonical edge types,
    in_feats=in_feats,
    hidden_feats=hidden_feats,
    out_feats=out_feats,
    num_layers=3,
    use_attention=True,
    predictor_type='mlp',
    target_etypes=target_etypes  # Only target edge types for prediction
)

print(f"Model created with {sum(p.numel() for p in model.parameters())} parameters")

embs = db.train_model(model, compute_loss, target_etypes, 'cuda', bs=500000, num_epochs=300)

# %%
torch.save(model, "/home/cc/PHD/dglframework/DeepKG/models/model-rgnn.pt")

# save_dict = {
#     'model_state_dict': model.state_dict(),
#     'model_class': model.__class__.__name__,
# }
# # Save the model
# torch.save(save_dict, "/home/cc/PHD/dglframework/DeepKG/models/model.pt")

# %%
from ChromaVDB.chroma import ChromaFramework
from tqdm.notebook import tqdm

vdb = ChromaFramework(persist_directory="./ChromaVDB/chroma_db")

# %%
from sentence_transformers import SentenceTransformer

# model = SentenceTransformer("all-MiniLM-L6-v2")
model = SentenceTransformer("all-mpnet-base-v2")

# %%
# 5 min x 130k nodes
BATCH_SIZE = 5000

for entity in db.graph.ntypes:
    embeddings_tensor = embs[entity].cpu()
    total = embeddings_tensor.shape[0]
    names = db.node_data[entity]['name'].tolist()

    text_embeddings = torch.tensor(model.encode(names), dtype=torch.float32)
    
    for i in tqdm(range(0, total, BATCH_SIZE)):
        end = i + BATCH_SIZE

        batch_ids = [db.reverse_node_mapping[(entity, j)] for j in range(i, min(end, total))]
        batch_embeddings = {"graph": embeddings_tensor[i:min(end, total)], "text": text_embeddings[i:min(end, total)]}
        batch_entities = [entity] * len(batch_embeddings["graph"])
        batch_names = names[i:min(end, total)]
        batch_metadata = [{} for _ in range(len(batch_embeddings["graph"]))]
        batch_docs = ["" for _ in range(len(batch_embeddings["graph"]))]

        vdb.create_records(
            global_ids=batch_ids,
            names=batch_names,
            entities=batch_entities,
            metadatas=batch_metadata,
            documents=batch_docs,
            embeddings=batch_embeddings
        )

# %%
records = vdb.list_records()
len(records)

# %%
import numpy as np
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
from sklearn.preprocessing import StandardScaler
import seaborn as sns

def visualize_embeddings_tsne(entity_embeddings_dict, 
                             perplexity=35, 
                             n_iter=1000, 
                             random_state=42,
                             figsize=(12, 8),
                             show_individual_labels=False,
                             font_size=8):
    """
    Create a t-SNE visualization of embeddings where each entity type has multiple embeddings.
    
    Parameters:
    entity_embeddings_dict (dict): Dictionary with entity types as keys and lists of embeddings as values
                                  e.g., {"person": [[emb1], [emb2], ...], "location": [[emb1], [emb2], ...]}
    perplexity (int): t-SNE perplexity parameter
    n_iter (int): Number of iterations for t-SNE
    random_state (int): Random state for reproducibility
    figsize (tuple): Figure size for the plot
    show_individual_labels (bool): Whether to show individual point labels (can be cluttered)
    font_size (int): Font size for labels
    """
    
    # Flatten the data and keep track of entity types
    all_embeddings = []
    entity_labels = []
    entity_types = []
    
    for entity_type, embeddings_list in entity_embeddings_dict.items():
        embeddings_array = np.array(embeddings_list.cpu())
        
        # Handle different input formats
        if embeddings_array.ndim == 1:
            embeddings_array = embeddings_array.reshape(1, -1)
        
        print(f"{entity_type}: {len(embeddings_array)} embeddings of dimension {embeddings_array.shape[1]}")
        
        for i, embedding in enumerate(embeddings_array):
            all_embeddings.append(embedding)
            entity_labels.append(f"{entity_type}_{i}")
            entity_types.append(entity_type)
    
    all_embeddings = np.array(all_embeddings)
    total_points = len(all_embeddings)
    
    print(f"\nTotal: {total_points} embeddings across {len(entity_embeddings_dict)} entity types")
    
    # Standardize the embeddings
    scaler = StandardScaler()
    embeddings_scaled = scaler.fit_transform(all_embeddings)
    
    # Apply t-SNE
    print("Applying t-SNE...")
    tsne = TSNE(n_components=2, 
                perplexity=min(perplexity, total_points-1),
                n_iter=n_iter, 
                random_state=random_state,
                verbose=1)
    
    embeddings_2d = tsne.fit_transform(embeddings_scaled)
    
    # Create the visualization
    plt.figure(figsize=figsize)
    
    # Create a color palette for entity types
    unique_entity_types = list(entity_embeddings_dict.keys())
    colors = sns.color_palette("husl", len(unique_entity_types))
    color_map = dict(zip(unique_entity_types, colors))
    
    # Plot points colored by entity type
    for entity_type in unique_entity_types:
        # Get indices for this entity type
        indices = [i for i, et in enumerate(entity_types) if et == entity_type]
        x_coords = embeddings_2d[indices, 0]
        y_coords = embeddings_2d[indices, 1]
        
        plt.scatter(x_coords, y_coords, 
                   c=[color_map[entity_type]], 
                   label=f"{entity_type} ({len(indices)})",
                   s=60, alpha=0.7, edgecolors='black', linewidth=0.5)
        
        # Optionally add individual labels
        if show_individual_labels:
            for i, idx in enumerate(indices):
                plt.annotate(f"{entity_type}_{i}", 
                            (embeddings_2d[idx, 0], embeddings_2d[idx, 1]),
                            xytext=(2, 2), textcoords='offset points',
                            fontsize=font_size, alpha=0.8)
    
    plt.title(f't-SNE Visualization of Entity Embeddings\n({total_points} total embeddings, {len(unique_entity_types)} entity types)', 
              fontsize=14, fontweight='bold')
    plt.xlabel('t-SNE Component 1', fontsize=12)
    plt.ylabel('t-SNE Component 2', fontsize=12)
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.grid(True, alpha=0.3)
    
    # Remove ticks for cleaner look
    plt.xticks([])
    plt.yticks([])
    
    plt.tight_layout()
    plt.show()
    
    return embeddings_2d, entity_labels, entity_types

def analyze_clusters(embeddings_2d, entity_types, entity_embeddings_dict):
    """
    Analyze the clustering quality and provide statistics.
    """
    from collections import Counter
    import pandas as pd
    
    # Create DataFrame for analysis
    df = pd.DataFrame({
        'x': embeddings_2d[:, 0],
        'y': embeddings_2d[:, 1],
        'entity_type': entity_types
    })
    
    print("\n=== Cluster Analysis ===")
    print("Entity type distribution:")
    type_counts = Counter(entity_types)
    for entity_type, count in type_counts.items():
        print(f"  {entity_type}: {count} embeddings")
    
    # Calculate centroids for each entity type
    print("\nEntity type centroids:")
    for entity_type in entity_embeddings_dict.keys():
        mask = df['entity_type'] == entity_type
        centroid_x = df[mask]['x'].mean()
        centroid_y = df[mask]['y'].mean()
        print(f"  {entity_type}: ({centroid_x:.3f}, {centroid_y:.3f})")
    
    return df

# Example usage with sample data
if __name__ == "__main__":
    # Example: Create sample embeddings dictionary with multiple embeddings per entity type
    np.random.seed(42)
    
    print("Running example with sample data...")
    print("Replace 'sample_entities' with your actual entity_embeddings_dict")
    
    # Visualize the embeddings
    tsne_coords, labels, types = visualize_embeddings_tsne(
        embs.cpu(),
        perplexity=35,  # Lower perplexity for smaller dataset
        figsize=(14, 10),
        show_individual_labels=False  # Set to True if you want individual point labels
    )
    
    # Analyze clusters
    # cluster_df = analyze_clusters(tsne_coords, types, sample_entities)
    
    # Optional: Save the plot
    # plt.savefig('entity_embeddings_tsne.png', dpi=300, bbox_inches='tight')