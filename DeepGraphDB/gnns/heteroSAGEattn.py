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

        print(node_types)
        
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
        
        self.dropout = nn.Dropout(0.3)
        
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

class RelationalGraphConvolutionalNetwork(nn.Module):
    """
    Relational Graph Convolutional Network (R-GCN) implementation
    """
    def __init__(self, node_types, edge_types, in_feats, hidden_feats, out_feats, num_layers=2):
        super().__init__()
        self.node_types = node_types
        self.edge_types = edge_types
        self.num_layers = num_layers

        # Input projection to map features of different node types to a common dimension
        self.input_proj = nn.ModuleDict({
            ntype: nn.Linear(in_feats[ntype] if isinstance(in_feats, dict) else in_feats, hidden_feats)
            for ntype in node_types
        })

        self.layers = nn.ModuleList()
        # R-GCN layers
        for i in range(num_layers):
            in_dim = hidden_feats
            out_dim = out_feats if i == num_layers - 1 else hidden_feats
            
            self.layers.append(dglnn.HeteroGraphConv({
                rel: dglnn.GraphConv(in_dim, out_dim)
                for rel in edge_types
            }, aggregate='mean'))

    def forward(self, blocks, x):
        h = {}
        # Apply input projection
        for ntype in x:
            h[ntype] = self.input_proj[ntype](x[ntype])

        # Propagate through layers
        for i, (layer, block) in enumerate(zip(self.layers, blocks)):
            h_new = layer(block, h)
            h = h_new
            if i < len(self.layers) - 1:
                h = {k: F.relu(v) for k, v in h.items()}
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

class AdvancedHeteroLinkPredictor(nn.Module):
    """
    Advanced heterogeneous link prediction model with multiple edge type support
    """
    def __init__(self, node_types, edge_types, canonical_etypes, in_feats, hidden_feats, out_feats, 
                 num_layers=3, use_attention=True, predictor_type='multi_edge', target_etypes=None):
        super().__init__()
        self.node_types = node_types
        self.edge_types = edge_types
        self.canonical_etypes = canonical_etypes
        self.target_etypes = target_etypes or edge_types
        
        # if gnn_type == 'sage':
        #     # Advanced GNN backbone (uses all edge types for message passing)
        #     self.gnn = HeteroGraphSAGE(
        #         node_types=node_types,
        #         edge_types=edge_types,
        #         in_feats=in_feats,
        #         hidden_feats=hidden_feats,
        #         out_feats=out_feats,
        #         num_layers=num_layers,
        #         use_attention=use_attention
        #     )
        # elif gnn_type == 'rgcn':
        #     self.gnn = RelationalGraphConvolutionalNetwork(
        #         node_types=node_types,
        #         edge_types=edge_types,
        #         in_feats=in_feats,
        #         hidden_feats=hidden_feats,
        #         out_feats=out_feats,
        #         num_layers=num_layers
        #     )
        # elif gnn_type == 'rgat':
        #     self.gnn = RelationalGATNetwork(
        #         node_types=node_types,
        #         edge_types=edge_types,
        #         in_feats=in_feats,
        #         hidden_feats=hidden_feats,
        #         out_feats=out_feats,
        #         num_layers=num_layers
        #     )
        # else:
        #     raise ValueError(f"Unsupported GNN type: {gnn_type}")

        self.gnn = RelationalGraphConvolutionalNetwork(
                node_types=node_types,
                edge_types=canonical_etypes,
                in_feats=in_feats,
                hidden_feats=hidden_feats,
                out_feats=out_feats,
                num_layers=num_layers
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
    
    def get_embeddings(self, graph, x):
        """
        Get node embeddings for the entire graph
        """
        h = self.gnn([graph, graph, graph], x)
        return {ntype: h[ntype].detach() for ntype in self.node_types}

# Enhanced loss function with margin-based ranking
#TODO: cuda esplode se la uso
def compute_margin_loss(pos_score, neg_score, margin=1.0):
    """
    Compute margin-based ranking loss
    """
    # Expand dimensions
    pos_score = pos_score.unsqueeze(1)  # [batch_size, 1]
    neg_score = neg_score.unsqueeze(0)  # [1, num_neg]
    
    # Compute margin loss 
    loss = torch.clamp(margin - pos_score + neg_score, min=0)
    # loss = torch.relu(margin - pos_score + neg_score)
    return loss.mean()

def compute_loss(pos_score, neg_score, loss_type='bce'):
    """
    Compute loss with different loss types
    """
    if loss_type == 'bce':
        pos_label = torch.ones_like(pos_score)
        neg_label = torch.zeros_like(neg_score)
        scores = torch.cat([pos_score, neg_score])
        labels = torch.cat([pos_label, neg_label])
        return F.binary_cross_entropy_with_logits(scores, labels)
    elif loss_type == 'margin':
        return compute_margin_loss(pos_score, neg_score)
    else:
        raise ValueError(f"Unknown loss type: {loss_type}")
