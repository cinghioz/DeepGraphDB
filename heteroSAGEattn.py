import os
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import dgl
import dgl.nn as dglnn
import dgl.function as fn
from dgl.dataloading import (
    DataLoader, 
    MultiLayerNeighborSampler, 
    as_edge_prediction_sampler
)
from sklearn.metrics import roc_auc_score, average_precision_score
import itertools
from typing import Dict, List, Tuple, Optional
from collections import defaultdict

# Set PyTorch backend for DGL
os.environ["DGLBACKEND"] = "pytorch"

class HeteroGraphSAGE(nn.Module):
    """
    Heterogeneous GraphSAGE implementation with attention mechanism
    """
    def __init__(self, node_types, edge_types, in_feats, hidden_feats, out_feats, 
                 num_layers=2, aggregator_type='mean', use_attention=True):
        super().__init__()
        self.node_types = node_types
        self.edge_types = edge_types
        self.num_layers = num_layers
        self.use_attention = use_attention
        
        # Input projection for different node types
        self.input_proj = nn.ModuleDict({
            ntype: nn.Linear(in_feats[ntype] if isinstance(in_feats, dict) else in_feats, hidden_feats)
            for ntype in node_types
        })
        
        # GraphSAGE layers
        self.sage_layers = nn.ModuleList()
        self.layer_norms = nn.ModuleList()
        
        for i in range(num_layers):
            in_dim = hidden_feats
            out_dim = hidden_feats if i < num_layers - 1 else out_feats
            
            # Use different aggregators for different layers
            conv_dict = {}
            for etype in edge_types:
                conv_dict[etype] = dglnn.SAGEConv(
                    in_feats=in_dim,
                    out_feats=out_dim,
                    aggregator_type=aggregator_type,
                    norm=F.relu if i < num_layers - 1 else None,
                    activation=F.relu if i < num_layers - 1 else None
                )
            
            self.sage_layers.append(
                dglnn.HeteroGraphConv(conv_dict, aggregate='sum')
            )
            
            # Layer norm for each layer with correct dimensions
            layer_norm_dict = nn.ModuleDict({
                ntype: nn.LayerNorm(out_dim) for ntype in node_types
            })
            self.layer_norms.append(layer_norm_dict)
        
        # Attention mechanism for heterogeneous message passing
        if use_attention:
            self.attention = nn.ModuleDict({
                ntype: nn.MultiheadAttention(
                    embed_dim=out_feats,
                    num_heads=4,
                    dropout=0.1,
                    batch_first=True
                ) for ntype in node_types
            })
        
        self.dropout = nn.Dropout(0.2)
        
    def forward(self, blocks, x):
        h = {}
        
        # Input projection
        for ntype in self.node_types:
            if ntype in x:
                h[ntype] = self.input_proj[ntype](x[ntype])
        
        # Forward through SAGE layers
        for i, (layer, block) in enumerate(zip(self.sage_layers, blocks)):
            h_new = layer(block, h)
            
            # Apply attention if enabled (only on final layer)
            if self.use_attention and i == len(self.sage_layers) - 1:
                for ntype in h_new:
                    if h_new[ntype].dim() == 2:
                        # Add sequence dimension for attention
                        h_input = h_new[ntype].unsqueeze(1)  # [N, 1, D]
                        attn_out, _ = self.attention[ntype](h_input, h_input, h_input)
                        h_new[ntype] = attn_out.squeeze(1)  # [N, D]
            
            # Layer normalization and residual connection
            if i > 0:  # Skip connection from previous layer
                for ntype in h_new:
                    if ntype in h and h[ntype].shape == h_new[ntype].shape:
                        h_new[ntype] = h_new[ntype] + h[ntype]
            
            # Apply layer norm and dropout with correct dimensions
            for ntype in h_new:
                h_new[ntype] = self.layer_norms[i][ntype](h_new[ntype])
                if i < len(self.sage_layers) - 1:
                    h_new[ntype] = self.dropout(h_new[ntype])
            
            h = h_new
        
        return h

class MultiEdgeTypePredictor(nn.Module):
    """
    Advanced predictor that can handle multiple edge types simultaneously
    """
    def __init__(self, in_features, hidden_features=128, edge_types=None):
        super().__init__()
        self.edge_types = edge_types or []
        
        # Shared feature transformation
        self.feature_transform = nn.Sequential(
            nn.Linear(in_features * 2, hidden_features),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.BatchNorm1d(hidden_features)
        )
        
        # Edge type specific predictors
        self.edge_predictors = nn.ModuleDict({
            f"{etype[0]}_{etype[1]}_{etype[2]}": nn.Sequential(
                nn.Linear(hidden_features, hidden_features // 2),
                nn.ReLU(),
                nn.Dropout(0.1),
                nn.Linear(hidden_features // 2, 1)
            ) for etype in self.edge_types
        })
        
        # Edge type embeddings for better prediction
        self.edge_type_embeddings = nn.Embedding(len(self.edge_types), hidden_features)
        
    def forward(self, graph, h, etype):
        with graph.local_scope():
            graph.ndata['h'] = h
            
            # Concatenate source and destination features
            src_type, edge_type, dst_type = etype
            src_nodes, dst_nodes = graph.edges(etype=etype)
            
            src_feat = h[src_type][src_nodes]
            dst_feat = h[dst_type][dst_nodes]
            edge_feat = torch.cat([src_feat, dst_feat], dim=1)
            
            # Transform features
            transformed_feat = self.feature_transform(edge_feat)
            
            # Add edge type embedding
            etype_key = f"{etype[0]}_{etype[1]}_{etype[2]}"
            if etype_key in self.edge_predictors and etype in self.edge_types:
                # Get edge type embedding
                etype_idx = torch.tensor([self.edge_types.index(etype)], device=edge_feat.device)
                etype_emb = self.edge_type_embeddings(etype_idx).expand(transformed_feat.size(0), -1)
                
                # Combine with edge features
                combined_feat = transformed_feat + etype_emb
                
                # Predict scores
                scores = self.edge_predictors[etype_key](combined_feat).squeeze()
            else:
                # Fallback to simple dot product
                scores = (src_feat * dst_feat).sum(dim=1)
            
            return scores

class AdvancedHeteroLinkPredictor(nn.Module):
    """
    Advanced heterogeneous link prediction model with multiple edge type support
    """
    def __init__(self, node_types, edge_types, in_feats, hidden_feats, out_feats, 
                 num_layers=3, use_attention=True, predictor_type='multi_edge', target_etypes=None):
        super().__init__()
        self.node_types = node_types
        self.edge_types = edge_types
        self.target_etypes = target_etypes or edge_types
        
        # Advanced GNN backbone (uses all edge types for message passing)
        self.gnn = HeteroGraphSAGE(
            node_types=node_types,
            edge_types=edge_types,
            in_feats=in_feats,
            hidden_feats=hidden_feats,
            out_feats=out_feats,
            num_layers=num_layers,
            use_attention=use_attention
        )
        
        # Advanced predictor (only for target edge types)
        if predictor_type == 'multi_edge':
            self.predictor = MultiEdgeTypePredictor(out_feats, hidden_feats, self.target_etypes)
        else:
            self.predictor = HeteroMLPPredictor(out_feats)
            
    def forward(self, pos_graph, neg_graph, blocks, x, etype):
        # Get node representations
        h = self.gnn(blocks, x)
        
        # Compute scores
        pos_score = self.predictor(pos_graph, h, etype)
        neg_score = self.predictor(neg_graph, h, etype)
        
        return pos_score, neg_score

class HeteroMLPPredictor(nn.Module):
    """Enhanced MLP predictor with residual connections"""
    def __init__(self, in_features, hidden_features=128):
        super().__init__()
        self.input_proj = nn.Linear(in_features * 2, hidden_features)
        
        # Residual blocks
        self.residual_blocks = nn.ModuleList([
            nn.Sequential(
                nn.Linear(hidden_features, hidden_features),
                nn.ReLU(),
                nn.Dropout(0.1),
                nn.Linear(hidden_features, hidden_features),
                nn.ReLU()
            ) for _ in range(2)
        ])
        
        self.output_proj = nn.Sequential(
            nn.Dropout(0.1),
            nn.Linear(hidden_features, 1)
        )
        
    def forward(self, graph, h, etype):
        with graph.local_scope():
            src_type, edge_type, dst_type = etype
            src_nodes, dst_nodes = graph.edges(etype=etype)
            
            src_feat = h[src_type][src_nodes]
            dst_feat = h[dst_type][dst_nodes]
            edge_feat = torch.cat([src_feat, dst_feat], dim=1)
            
            # Input projection
            x = self.input_proj(edge_feat)
            
            # Residual blocks
            for block in self.residual_blocks:
                residual = x
                x = block(x) + residual
            
            # Output projection
            scores = self.output_proj(x).squeeze()
            return scores