import os
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import dgl
import dgl.nn as dglnn
from dgl.dataloading import DataLoader, MultiLayerNeighborSampler
from sklearn.metrics import roc_auc_score, average_precision_score
import copy
from typing import Dict, List, Tuple, Optional, Union
from collections import defaultdict
import logging

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class PatientSpecificGNNAdapter(nn.Module):
    """
    Adapter module for patient-specific fine-tuning
    Uses low-rank adaptation (LoRA) approach to minimize overfitting
    """
    def __init__(self, original_dim, rank=16, alpha=16):
        super().__init__()
        self.rank = rank
        self.alpha = alpha
        self.scaling = alpha / rank
        
        # Low-rank matrices for adaptation
        self.lora_A = nn.Linear(original_dim, rank, bias=False)
        self.lora_B = nn.Linear(rank, original_dim, bias=False)
        
        # Initialize weights
        nn.init.kaiming_uniform_(self.lora_A.weight, a=np.sqrt(5))
        nn.init.zeros_(self.lora_B.weight)
        
    def forward(self, x):
        return self.lora_B(self.lora_A(x)) * self.scaling

class PatientAdaptiveGNN(nn.Module):
    """
    Patient-adaptive version of the heterogeneous GNN with LoRA adapters
    """
    def __init__(self, base_model, node_types, adapter_rank=16, adapter_alpha=16):
        super().__init__()
        self.base_model = base_model
        self.node_types = node_types
        self.use_adapters = True
        
        # Freeze base model parameters
        for param in self.base_model.parameters():
            param.requires_grad = False
        
        # Add LoRA adapters to each node type's input projection
        self.adapters = nn.ModuleDict({
            ntype: PatientSpecificGNNAdapter(
                original_dim=self.base_model.gnn.input_proj[ntype].in_features,
                rank=adapter_rank,
                alpha=adapter_alpha
            ) for ntype in node_types
        })
        
        # Patient-specific normalization layers
        self.patient_norms = nn.ModuleDict({
            ntype: nn.LayerNorm(self.base_model.gnn.input_proj[ntype].out_features)
            for ntype in node_types
        })
        
    def forward(self, pos_graph, neg_graph, blocks, x, etype):
        # Apply adapters to input features
        adapted_x = {}
        for ntype in self.node_types:
            if ntype in x:
                base_feat = self.base_model.gnn.input_proj[ntype](x[ntype])
                if self.use_adapters:
                    adapter_feat = self.adapters[ntype](x[ntype])
                    adapted_feat = base_feat + adapter_feat
                else:
                    adapted_feat = base_feat
                adapted_x[ntype] = self.patient_norms[ntype](adapted_feat)
        
        # Forward through the rest of the base model
        h = adapted_x
        for i, (layer, block) in enumerate(zip(self.base_model.gnn.sage_layers, blocks)):
            h_new = layer(block, h)
            
            # Apply attention if enabled (only on final layer)
            if self.base_model.gnn.use_attention and i == len(self.base_model.gnn.sage_layers) - 1:
                for ntype in h_new:
                    if h_new[ntype].dim() == 2:
                        h_input = h_new[ntype].unsqueeze(1)
                        attn_out, _ = self.base_model.gnn.attention[ntype](h_input, h_input, h_input)
                        h_new[ntype] = attn_out.squeeze(1)
            
            # Layer normalization and residual connection
            if i > 0:
                for ntype in h_new:
                    if ntype in h and h[ntype].shape == h_new[ntype].shape:
                        h_new[ntype] = h_new[ntype] + h[ntype]
            
            for ntype in h_new:
                h_new[ntype] = self.base_model.gnn.layer_norms[i][ntype](h_new[ntype])
                if i < len(self.base_model.gnn.sage_layers) - 1:
                    h_new[ntype] = self.base_model.gnn.dropout(h_new[ntype])
            
            h = h_new
        
        # Compute scores using base model predictor
        pos_score = self.base_model.predictor(pos_graph, h, etype)
        neg_score = self.base_model.predictor(neg_graph, h, etype)
        
        return pos_score, neg_score

class PatientSpecificFineTuner:
    """
    Fine-tuning manager for patient-specific GNN adaptation
    """
    def __init__(self, base_model_path, node_types, edge_types, device='cuda'):
        self.device = device
        self.node_types = node_types
        self.edge_types = edge_types
        
        # Load pretrained base model
        self.base_model = torch.load(base_model_path, map_location=device)
        self.base_model.eval()
        
        logger.info("Loaded pretrained base model")
        
    def create_patient_model(self, patient_id, adapter_rank=16, adapter_alpha=16):
        """
        Create a patient-specific model with adapters
        """
        patient_model = PatientAdaptiveGNN(
            base_model=self.base_model,
            node_types=self.node_types,
            adapter_rank=adapter_rank,
            adapter_alpha=adapter_alpha
        ).to(self.device)
        
        logger.info(f"Created patient-specific model for patient {patient_id}")
        return patient_model
    
    def prepare_patient_data(self, patient_graph, target_etypes, train_ratio=0.8, neg_sampling_ratio=5):
        """
        Prepare training data for a specific patient
        """
        train_edges = {}
        val_edges = {}
        
        logger.info(f"Patient graph has {patient_graph.number_of_nodes()} nodes and {patient_graph.number_of_edges()} edges")
        logger.info(f"Available edge types: {patient_graph.etypes}")
        
        for etype in target_etypes:
            if etype in patient_graph.etypes:
                # Get all edges of this type
                src, dst = patient_graph.edges(etype=etype)
                num_edges = len(src)
                
                logger.info(f"Edge type {etype}: {num_edges} edges")
                
                if num_edges > 0:
                    # Ensure we have enough edges for train/val split
                    if num_edges < 2:
                        logger.warning(f"Too few edges for {etype} ({num_edges}), using all for training")
                        train_edges[etype] = (src, dst)
                        val_edges[etype] = (src[:0], dst[:0])  # Empty validation set
                    else:
                        # Split edges into train/val
                        perm = torch.randperm(num_edges)
                        train_size = max(1, int(train_ratio * num_edges))  # At least 1 for training
                        
                        train_idx = perm[:train_size]
                        val_idx = perm[train_size:]
                        
                        train_edges[etype] = (src[train_idx], dst[train_idx])
                        val_edges[etype] = (src[val_idx], dst[val_idx])
                        
                        logger.info(f"  Train: {len(train_idx)} edges, Val: {len(val_idx)} edges")
            else:
                logger.warning(f"Edge type {etype} not found in patient graph")
        
        return train_edges, val_edges
    
    def fine_tune_patient_model(self, patient_model, patient_graph, train_edges, val_edges, 
                              target_etypes, num_epochs=50, lr=1e-4, weight_decay=1e-5,
                              early_stopping_patience=10):
        """
        Fine-tune model for a specific patient
        """
        # Setup optimizer (only train adapter parameters)
        trainable_params = []
        for name, param in patient_model.named_parameters():
            if param.requires_grad:
                trainable_params.append(param)
        
        logger.info(f"Training {len(trainable_params)} adapter parameters")
        
        optimizer = torch.optim.AdamW(trainable_params, lr=lr, weight_decay=weight_decay)
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='max', patience=5, factor=0.5)
        
        best_val_auc = 0
        patience_counter = 0
        best_model_state = copy.deepcopy(patient_model.state_dict())  # Initialize here
        
        # Prepare input features once
        input_features = {}
        for ntype in patient_model.node_types:
            if patient_graph.number_of_nodes(ntype) > 0:
                if 'feat' in patient_graph.nodes[ntype].data:
                    input_features[ntype] = patient_graph.nodes[ntype].data['feat'].to(self.device)
                else:
                    # Initialize with random features
                    feat_dim = patient_model.base_model.gnn.input_proj[ntype].in_features
                    input_features[ntype] = torch.randn(
                        patient_graph.number_of_nodes(ntype), feat_dim
                    ).to(self.device)
        
        for epoch in range(num_epochs):
            patient_model.train()
            total_loss = 0
            num_batches = 0
            
            # Training loop
            for etype in target_etypes:
                if etype not in train_edges:
                    continue
                    
                train_src, train_dst = train_edges[etype]
                if len(train_src) == 0:
                    continue
                
                # Create positive edges graph
                pos_edges = {etype: (train_src, train_dst)}
                pos_graph = dgl.graph(pos_edges, num_nodes_dict={
                    ntype: patient_graph.number_of_nodes(ntype) for ntype in patient_graph.ntypes
                }).to(self.device)
                
                # Negative sampling - sample from the original graph structure
                neg_src, neg_dst = dgl.sampling.global_uniform_negative_sampling(
                    patient_graph, len(train_src) * 3, etype=etype
                )
                
                if len(neg_src) == 0:
                    logger.warning(f"No negative samples for {etype}")
                    continue
                
                # Create negative edges graph  
                neg_edges = {etype: (neg_src, neg_dst)}
                neg_graph = dgl.graph(neg_edges, num_nodes_dict={
                    ntype: patient_graph.number_of_nodes(ntype) for ntype in patient_graph.ntypes
                }).to(self.device)
                
                # For simplicity, use the full graph as "blocks" - you may want to implement proper sampling
                blocks = [patient_graph.to(self.device)] * patient_model.base_model.gnn.num_layers
                
                try:
                    # Forward pass
                    pos_score, neg_score = patient_model(pos_graph, neg_graph, blocks, input_features, etype)
                    
                    # Check if scores are valid
                    if torch.isnan(pos_score).any() or torch.isnan(neg_score).any():
                        logger.warning(f"NaN scores detected for {etype}")
                        continue
                    
                    # Compute loss
                    loss = self.compute_loss(pos_score, neg_score)
                    
                    if torch.isnan(loss):
                        logger.warning(f"NaN loss detected for {etype}")
                        continue
                    
                    # Backward pass
                    optimizer.zero_grad()
                    loss.backward()
                    torch.nn.utils.clip_grad_norm_(trainable_params, max_norm=1.0)
                    optimizer.step()
                    
                    total_loss += loss.item()
                    num_batches += 1
                    
                except Exception as e:
                    logger.warning(f"Error processing {etype}: {e}")
                    continue
            
            avg_loss = total_loss / max(num_batches, 1)
            
            # Validation every epoch to debug
            val_auc = self.evaluate_patient_model(patient_model, patient_graph, val_edges, target_etypes)
            
            if epoch % 5 == 0 or epoch < 5:
                logger.info(f"Epoch {epoch}: Loss={avg_loss:.4f}, Val AUC={val_auc:.4f}, Batches={num_batches}")
            
            # Early stopping
            if val_auc > best_val_auc:
                best_val_auc = val_auc
                patience_counter = 0
                best_model_state = copy.deepcopy(patient_model.state_dict())
            else:
                patience_counter += 1
            
            scheduler.step(val_auc)
            
            if patience_counter >= early_stopping_patience and epoch > 10:
                logger.info(f"Early stopping at epoch {epoch}")
                break
        
        # Load best model state
        patient_model.load_state_dict(best_model_state)
        return patient_model, best_val_auc
    
    def evaluate_patient_model(self, patient_model, patient_graph, val_edges, target_etypes):
        """
        Evaluate patient model on validation data
        """
        patient_model.eval()
        all_scores = []
        all_labels = []
        
        # Prepare input features once
        input_features = {}
        for ntype in patient_model.node_types:
            if patient_graph.number_of_nodes(ntype) > 0:
                if 'feat' in patient_graph.nodes[ntype].data:
                    input_features[ntype] = patient_graph.nodes[ntype].data['feat'].to(self.device)
                else:
                    feat_dim = patient_model.base_model.gnn.input_proj[ntype].in_features
                    input_features[ntype] = torch.randn(
                        patient_graph.number_of_nodes(ntype), feat_dim
                    ).to(self.device)
        
        with torch.no_grad():
            for etype in target_etypes:
                if etype not in val_edges:
                    continue
                
                val_src, val_dst = val_edges[etype]
                if len(val_src) == 0:
                    continue
                
                try:
                    # Create positive edges graph
                    pos_edges = {etype: (val_src, val_dst)}
                    pos_graph = dgl.graph(pos_edges, num_nodes_dict={
                        ntype: patient_graph.number_of_nodes(ntype) for ntype in patient_graph.ntypes
                    }).to(self.device)
                    
                    # Negative sampling for evaluation
                    neg_src, neg_dst = dgl.sampling.global_uniform_negative_sampling(
                        patient_graph, len(val_src), etype=etype
                    )
                    
                    if len(neg_src) == 0:
                        continue
                    
                    # Create negative edges graph
                    neg_edges = {etype: (neg_src, neg_dst)}
                    neg_graph = dgl.graph(neg_edges, num_nodes_dict={
                        ntype: patient_graph.number_of_nodes(ntype) for ntype in patient_graph.ntypes
                    }).to(self.device)
                    
                    # Use full graph as blocks for simplicity
                    blocks = [patient_graph.to(self.device)] * patient_model.base_model.gnn.num_layers
                    
                    # Forward pass
                    pos_score, neg_score = patient_model(pos_graph, neg_graph, blocks, input_features, etype)
                    
                    # Check for valid scores
                    if torch.isnan(pos_score).any() or torch.isnan(neg_score).any():
                        logger.warning(f"NaN scores in evaluation for {etype}")
                        continue
                    
                    # Collect scores and labels
                    pos_scores_np = torch.sigmoid(pos_score).cpu().numpy()
                    neg_scores_np = torch.sigmoid(neg_score).cpu().numpy()
                    
                    scores = np.concatenate([pos_scores_np, neg_scores_np])
                    labels = np.concatenate([np.ones(len(pos_scores_np)), np.zeros(len(neg_scores_np))])
                    
                    all_scores.extend(scores)
                    all_labels.extend(labels)
                    
                except Exception as e:
                    logger.warning(f"Error in evaluation for {etype}: {e}")
                    continue
        
        if len(all_scores) > 0 and len(set(all_labels)) > 1:
            try:
                auc = roc_auc_score(all_labels, all_scores)
                return auc
            except Exception as e:
                logger.warning(f"Error computing AUC: {e}")
                return 0.0
        else:
            logger.warning("No valid scores for AUC computation or labels are all the same")
            return 0.0
    
    def compute_loss(self, pos_score, neg_score, loss_type='bce'):
        """
        Compute loss with regularization for fine-tuning
        """
        if loss_type == 'bce':
            pos_label = torch.ones_like(pos_score)
            neg_label = torch.zeros_like(neg_score)
            scores = torch.cat([pos_score, neg_score])
            labels = torch.cat([pos_label, neg_label])
            return F.binary_cross_entropy_with_logits(scores, labels)
        else:
            # Margin loss
            margin = 1.0
            loss = torch.clamp(margin - pos_score.mean() + neg_score.mean(), min=0)
            return loss
    
    def batch_fine_tune_patients(self, patient_data_dict, target_etypes, save_dir='patient_models'):
        """
        Fine-tune models for multiple patients
        
        Args:
            patient_data_dict: Dict[str, dgl.DGLGraph] - patient_id -> patient_graph
            target_etypes: List of edge types to predict
            save_dir: Directory to save patient models
        """
        os.makedirs(save_dir, exist_ok=True)
        results = {}
        
        for patient_id, patient_graph in patient_data_dict.items():
            logger.info(f"Fine-tuning model for patient {patient_id}")
            
            # Create patient-specific model
            patient_model = self.create_patient_model(patient_id)
            
            # Prepare data
            train_edges, val_edges = self.prepare_patient_data(patient_graph, target_etypes)
            
            # Fine-tune
            fine_tuned_model, best_auc = self.fine_tune_patient_model(
                patient_model, patient_graph, train_edges, val_edges, target_etypes
            )
            
            # Save model
            model_path = os.path.join(save_dir, f'patient_{patient_id}_model.pt')
            torch.save(fine_tuned_model.state_dict(), model_path)
            
            results[patient_id] = {
                'model_path': model_path,
                'best_auc': best_auc,
                'num_train_edges': sum(len(edges[0]) for edges in train_edges.values()),
                'num_val_edges': sum(len(edges[0]) for edges in val_edges.values())
            }
            
            logger.info(f"Patient {patient_id}: Best AUC = {best_auc:.4f}")
        
    def debug_patient_data(self, patient_graph, target_etypes):
        """
        Debug function to check patient data quality
        """
        logger.info("=== DEBUGGING PATIENT DATA ===")
        logger.info(f"Graph nodes: {patient_graph.number_of_nodes()}")
        logger.info(f"Graph edges: {patient_graph.number_of_edges()}")
        logger.info(f"Node types: {patient_graph.ntypes}")
        logger.info(f"Edge types: {patient_graph.etypes}")
        
        for ntype in patient_graph.ntypes:
            num_nodes = patient_graph.number_of_nodes(ntype)
            logger.info(f"  {ntype}: {num_nodes} nodes")
            if 'feat' in patient_graph.nodes[ntype].data:
                feat_shape = patient_graph.nodes[ntype].data['feat'].shape
                logger.info(f"    Features: {feat_shape}")
            else:
                logger.info(f"    No features available")
        
        for etype in patient_graph.etypes:
            num_edges = patient_graph.number_of_edges(etype)
            logger.info(f"  {etype}: {num_edges} edges")
            
        logger.info(f"Target edge types: {target_etypes}")
        
        # Check if target edge types exist
        missing_etypes = [etype for etype in target_etypes if etype not in patient_graph.etypes]
        if missing_etypes:
            logger.error(f"Missing target edge types: {missing_etypes}")
            
        return len(missing_etypes) == 0