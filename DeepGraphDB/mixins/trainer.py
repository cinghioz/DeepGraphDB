import dgl
import torch
from typing import Dict, List, Set, Tuple, Any
from collections import defaultdict
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score, average_precision_score
import logging
import os
from tqdm import tqdm

logger = logging.getLogger(__name__)

class TrainerMixin:
    def split_edges_consistent(self, target_etypes, train_ratio=0.70, val_ratio=0.15, test_ratio=0.15, random_state=42):
        """
        Split edges consistently across different edge types in a heterogeneous graph.
        
        Args:
            graph: DGL heterogeneous graph
            target_etypes: List of edge types to split
            train_ratio: Ratio for training set
            val_ratio: Ratio for validation set  
            test_ratio: Ratio for test set
            random_state: Random seed for reproducibility
        
        Returns:
            Dictionary containing train/val/test edge indices for each edge type
        """
        assert abs(train_ratio + val_ratio + test_ratio - 1.0) < 1e-6, "Ratios must sum to 1.0"
        
        edge_splits = {}
        np.random.seed(random_state)
        
        for etype in target_etypes:
            src, dst = self.graph.edges(etype=etype)
            num_edges = len(src)
            
            if num_edges == 0:
                edge_splits[etype] = {
                    'train': {'src': torch.tensor([]), 'dst': torch.tensor([])},
                    'val': {'src': torch.tensor([]), 'dst': torch.tensor([])},
                    'test': {'src': torch.tensor([]), 'dst': torch.tensor([])}
                }
                continue
            
            # Create edge indices
            edge_indices = np.arange(num_edges)
            
            # First split: train vs (val + test)
            train_indices, temp_indices = train_test_split(
                edge_indices, 
                train_size=train_ratio, 
                random_state=random_state
            )
            
            # Second split: val vs test from the remaining
            val_size = val_ratio / (val_ratio + test_ratio)
            val_indices, test_indices = train_test_split(
                temp_indices, 
                train_size=val_size, 
                random_state=random_state
            )
            
            # Store the splits
            edge_splits[etype] = {
                'train': {
                    'src': src[train_indices],
                    'dst': dst[train_indices]
                },
                'val': {
                    'src': src[val_indices], 
                    'dst': dst[val_indices]
                },
                'test': {
                    'src': src[test_indices],
                    'dst': dst[test_indices]
                }
            }
            
            print(f"Edge type {etype}: {len(train_indices)} train, {len(val_indices)} val, {len(test_indices)} test")
    
        return edge_splits

    def create_train_graph(self, edge_splits, target_etypes):
        """
        Create a training graph that only contains training edges.
        """
        train_edge_dict = {}
        
        # Add all non-target edge types (keep full connectivity for message passing)
        for etype in self.graph.canonical_etypes:
            if etype not in target_etypes:
                src, dst = self.graph.edges(etype=etype)
                train_edge_dict[etype] = (src, dst)
            else:
                # Only add training edges for target edge types
                train_edges = edge_splits[etype]['train']
                if len(train_edges['src']) > 0:
                    train_edge_dict[etype] = (train_edges['src'], train_edges['dst'])
        
        # Create training graph
        train_graph = dgl.heterograph(
            train_edge_dict,
            num_nodes_dict={ntype: self.graph.num_nodes(ntype) for ntype in self.graph.ntypes}
        )
        
        # Copy node features
        for ntype in self.graph.ntypes:
            if self.graph.nodes[ntype].data:
                for key, value in self.graph.nodes[ntype].data.items():
                    train_graph.nodes[ntype].data[key] = value
        
        return train_graph

    def advanced_negative_sampling(self, graph, etype, k=1, method='uniform'):
        """
        Advanced negative sampling with different strategies
        """
        if method == 'uniform':
            return self.negative_sampling(graph, etype, k)
        elif method == 'popularity_based':
            # Sample negative examples based on node popularity
            src_type, _, dst_type = etype
            src, dst = graph.edges(etype=etype)
            
            # Calculate node degrees for popularity-based sampling
            try:
                dst_degrees = graph.in_degrees(graph.nodes(dst_type), etype=etype).float()
            except:
                # Fallback to uniform sampling if degree calculation fails
                return self.negative_sampling(graph, etype, k)
                
            if dst_degrees.sum() == 0:
                # Fallback to uniform sampling if no degrees
                return self.negative_sampling(graph, etype, k)
                
            dst_probs = dst_degrees / dst_degrees.sum()
            
            # Sample negative destinations based on popularity
            neg_dst = torch.multinomial(dst_probs, len(src) * k, replacement=True)
            neg_src = src.repeat_interleave(k)
            
            # Create negative graph
            neg_graph = dgl.heterograph(
                {etype: (neg_src, neg_dst)},
                num_nodes_dict={ntype: graph.num_nodes(ntype) for ntype in graph.ntypes}
            )
            return neg_graph
        else:
            raise ValueError(f"Unknown negative sampling method: {method}")

    # def negative_sampling(self, graph, etype, k=1):
    #     """
    #     Standard negative sampling implementation using manual sampling
    #     """
    #     src_type, _, dst_type = etype
    #     src, dst = graph.edges(etype=etype)
        
    #     # Manual negative sampling
    #     num_pos_edges = len(src)
    #     num_neg_edges = num_pos_edges * k
        
    #     # Sample negative source nodes (same as positive)
    #     neg_src = src.repeat_interleave(k)
        
    #     # Sample negative destination nodes uniformly
    #     num_dst_nodes = graph.num_nodes(dst_type)
    #     neg_dst = torch.randint(0, num_dst_nodes, (num_neg_edges,), device=src.device)
        
    #     # Create negative graph
    #     neg_graph = dgl.heterograph(
    #         {etype: (neg_src, neg_dst)},
    #         num_nodes_dict={ntype: graph.num_nodes(ntype) for ntype in graph.ntypes}
    #     )
        
    #     return neg_graph

    def negative_sampling(self, graph, etype, k=1):
        """
        Negative sampling implementation that selects destination nodes
        that do not appear in the real destination tensor.
        """
        src_type, _, dst_type = etype
        src, dst = graph.edges(etype=etype)
        
        # Get all unique destination nodes from the positive edges
        unique_dst = torch.unique(dst)
        num_dst_nodes = graph.num_nodes(dst_type)
        
        # Create a boolean mask to identify nodes that are NOT in the unique_dst tensor
        # is_candidate will be True for nodes that can be used as negative samples
        is_candidate = torch.ones(num_dst_nodes, dtype=torch.bool, device=src.device)
        is_candidate[unique_dst] = False
        
        # Get the tensor of candidate destination nodes
        candidate_neg_dst = torch.arange(num_dst_nodes, device=src.device)[is_candidate]
        
        num_pos_edges = len(src)
        num_neg_edges = num_pos_edges * k
        
        # Sample negative source nodes (same as positive)
        neg_src = src.repeat_interleave(k)
        
        # Check if there are any candidate nodes to sample from
        if len(candidate_neg_dst) > 0:
            # Sample with replacement from the candidate negative destination nodes
            neg_dst_indices = torch.randint(0, len(candidate_neg_dst), (num_neg_edges,), device=src.device)
            neg_dst = candidate_neg_dst[neg_dst_indices]
        else:
            # Fallback: If all nodes are destination nodes in the graph,
            # revert to uniform random sampling.
            neg_dst = torch.randint(0, num_dst_nodes, (num_neg_edges,), device=src.device)
            
        # Create negative graph
        neg_graph = dgl.heterograph(
            {etype: (neg_src, neg_dst)},
            num_nodes_dict={ntype: graph.num_nodes(ntype) for ntype in graph.ntypes}
        )
        
        return neg_graph

    def multi_metric_evaluation(self, pos_scores, neg_scores):
        """
        Compute multiple evaluation metrics
        """
        scores = torch.cat([pos_scores, neg_scores]).detach().cpu().numpy()
        labels = torch.cat([
            torch.ones(pos_scores.shape[0]),
            torch.zeros(neg_scores.shape[0])
        ]).cpu().numpy()
        
        auc = roc_auc_score(labels, scores)
        ap = average_precision_score(labels, scores)
        
        # Compute Hit@K metrics
        k_values = [1, 5, 10]
        hit_at_k = {}
        
        for k in k_values:
            # Sort scores in descending order
            sorted_indices = np.argsort(scores)[::-1]
            top_k_labels = labels[sorted_indices[:k]]
            hit_at_k[f'Hit@{k}'] = np.sum(top_k_labels) / min(k, np.sum(labels))
        
        return {
            'AUC': auc,
            'AP': ap,
            **hit_at_k
        }

    def evaluate_on_split(self, model, edge_splits, target_etypes, split_name, device, num_neg_samples=5):
        """
        Evaluate model on a specific data split (val or test) using multiple negative samples.
        
        Args:
            model: The trained model
            graph: Full graph for message passing
            edge_splits: Dictionary containing edge splits
            target_etypes: List of target edge types
            split_name: 'val' or 'test'
            device: Device to run evaluation on
            num_neg_samples: Number of different negative graphs to average over
        
        Returns:
            split_metrics: Dictionary with averaged metrics
            metrics_std: Dictionary with standard deviations (only returned for final evaluation)
        """
        model.eval()
        split_metrics = defaultdict(list)
        all_metrics = defaultdict(lambda: defaultdict(list)) 
        
        with torch.no_grad():
            for target_etype in target_etypes:
                split_edges = edge_splits[target_etype][split_name]
                
                if len(split_edges['src']) == 0:
                    continue
                
                # Create positive graph for evaluation
                pos_graph = dgl.heterograph(
                    {target_etype: (split_edges['src'], split_edges['dst'])},
                    num_nodes_dict={ntype: self.graph.num_nodes(ntype) for ntype in self.graph.ntypes}
                ).to(device)
                
                # Prepare input features
                input_features = {ntype: self.graph.nodes[ntype].data['x'] for ntype in self.graph.ntypes}
                
                # Create blocks for GNN (using full graph for message passing)
                blocks = [self.graph, self.graph, self.graph]
                
                # Evaluate with multiple negative samples
                etype_metrics = defaultdict(list)
                
                for neg_sample_idx in range(num_neg_samples):
                    # Generate different negative samples for each iteration
                    neg_graph = self.advanced_negative_sampling(pos_graph, target_etype, k=1, method='uniform').to(device)
                    
                    # Forward pass
                    pos_score, neg_score = model(pos_graph, neg_graph, blocks, input_features, target_etype)
                    
                    # Compute metrics for this negative sample
                    metrics = self.multi_metric_evaluation(pos_score, neg_score)
                    for metric_name, value in metrics.items():
                        etype_metrics[metric_name].append(value)
                        all_metrics[target_etype][metric_name].append(value)
                
                # Average metrics across all negative samples
                for metric_name, values in etype_metrics.items():
                    avg_metric = np.mean(values)
                    split_metrics[f"{target_etype}_{metric_name}"].append(avg_metric)
        
        # Calculate standard deviations for detailed evaluation
        metrics_std = {}
        for etype in target_etypes:
            for metric_name in all_metrics[etype]:
                std_key = f"{etype}_{metric_name}_std"
                metrics_std[std_key] = np.std(all_metrics[etype][metric_name])

        return split_metrics, metrics_std

    def train_model(self, model, loss_f, target_etypes, target_entities, device, bs=500000, num_epochs=100):
        self.move_to_device(device)
        model = model.to(device)

        print("\nSplitting edges into train/val/test sets...")
        edge_splits = self.split_edges_consistent(
            target_etypes, 
            train_ratio=0.70, 
            val_ratio=0.15, 
            test_ratio=0.15,
            random_state=42
        )

        # Create training graph (only contains training edges for target edge types)
        train_graph = self.create_train_graph(edge_splits, target_etypes)
        print(f"Training graph created with {train_graph.num_edges()} edges")

        train_graph = train_graph.to(device)

        # Training configuration
        optimizer = torch.optim.AdamW(model.parameters(), lr=0.001, weight_decay=1e-5)
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=50, factor=0.5)

        print("Starting training with proper train/val splits...")
        best_val_metrics = { etype: 0.0 for etype in target_etypes }
        patience_counter = 0
        max_patience = 50

        for epoch in tqdm(range(num_epochs)):
            model.train()
            total_loss = 0
            
            # Train on each edge type using only training edges
            for target_etype in target_etypes:
                train_edges = edge_splits[target_etype]['train']
                
                if len(train_edges['src']) == 0:
                    continue
                
                src, dst = train_edges['src'], train_edges['dst']
                
                # Create mini-batches
                batch_size = min(bs, len(src))
                num_batches = (len(src) + batch_size - 1) // batch_size
                
                # TODO: fixxare il dataloder. Questo è un workaround per evitare errori di memoria

                for batch_idx in range(num_batches):
                    start_idx = batch_idx * batch_size
                    end_idx = min((batch_idx + 1) * batch_size, len(src))
                    
                    batch_src = src[start_idx:end_idx]
                    batch_dst = dst[start_idx:end_idx]
                    
                    # Create positive graph for batch
                    pos_graph = dgl.heterograph(
                        {target_etype: (batch_src, batch_dst)},
                        num_nodes_dict={ntype: self.graph.num_nodes(ntype) for ntype in self.graph.ntypes}
                    ).to(device)
                    
                    # Generate negative samples
                    neg_graph = self.advanced_negative_sampling(
                        pos_graph, target_etype, k=1, method='uniform'
                    ).to(device)
                    
                    # Prepare input features (use full graph features)
                    input_features = {ntype: self.graph.nodes[ntype].data['x'] for ntype in self.graph.ntypes}
                    
                    # Create blocks for GNN (use full graph for message passing)
                    blocks = [self.graph, self.graph, self.graph]
                    
                    # Forward pass
                    pos_score, neg_score = model(pos_graph, neg_graph, blocks, input_features, target_etype)
                    
                    del pos_graph, neg_graph, blocks  # Free memory
                    # torch.cuda.empty_cache()  # Clear cache to avoid memory issues

                    # Compute loss
                    loss = loss_f(pos_score, neg_score)
                    
                    # Backward pass
                    optimizer.zero_grad()
                    loss.backward()
                    torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                    optimizer.step()
                    
                    total_loss += loss.item()
            
            if (epoch+1) % 30 == 0:  # Evaluate every 30 epochs
                # Evaluate on validation set (val on 10 different negative graph)
                val_metrics, _ = self.evaluate_on_split(model, edge_splits, target_etypes, 'val', device, 10)
                
                print(f"\nEpoch {epoch:03d} | Loss: {total_loss:.4f}")
                
                # Track best validation metrics and calculate average
                epoch_improved = True
                epoch_aucs = []
                epoch_aps = []

                for etype in target_etypes:
                    if f"{etype}_AUC" in val_metrics and f"{etype}_AP" in val_metrics:
                        auc_values = val_metrics[f"{etype}_AUC"]
                        ap_values = val_metrics[f"{etype}_AP"]

                        if auc_values and ap_values:
                            avg_auc = np.mean(auc_values)
                            avg_ap = np.mean(ap_values)
                            epoch_aucs.append(avg_auc)
                            epoch_aps.append(avg_ap)

                            print(f"  {etype} Val AUC: {avg_auc:.4f}, AP: {avg_ap:.4f}")
                            
                            if avg_auc > best_val_metrics[etype]:
                                best_val_metrics[etype] = avg_auc
                                epoch_improved = True
                
                # Print average AUC across all edge types
                if epoch_aucs:
                    avg_auc_all = np.mean(epoch_aucs)
                    print(f"  Average Val AUC: {avg_auc_all:.4f}")

                if epoch_aps:
                    avg_ap_all = np.mean(epoch_aps)
                    print(f"  Average Val AP: {avg_ap_all:.4f}")
                
                # Early stopping logic
                if epoch_improved:
                    patience_counter = 0
                else:
                    patience_counter += 1
                    
                if patience_counter >= max_patience:
                    print(f"\nEarly stopping at epoch {epoch} (no improvement for {max_patience} evaluations)")
                    break
            
            # Scheduler step
            scheduler.step(total_loss)

        print("\nTraining completed!")

        self._final_evaluation(model, edge_splits, target_etypes, device)

        embs = model.get_embeddings(self.graph, {ntype: self.graph.nodes[ntype].data['x'] for ntype in self.graph.ntypes})

        if not os.path.exists('models'):
            os.makedirs('models')

        if not os.path.exists('embeddings'):
            os.makedirs('embeddings')

        # Save the model and embeddings
        torch.save(model.state_dict(), 'models/model.pth')
        print("Model saved as 'model.pth'")
        
        for ntype, emb in embs.items():
            torch.save(emb, f'embeddings/embeddings_{ntype}.pth')
            print(f"Embeddings for {ntype} saved as 'embeddings_{ntype}.pth'")

        print("Embeddings saved successfully.")

        return embs

    def _final_evaluation(self, model, edge_splits, target_etypes, device):
        print("\n" + "="*50)
        print("FINAL EVALUATION")
        print("="*50)

        # Validation set evaluation
        print("\nValidation Set Results:")
        val_metrics, _ = self.evaluate_on_split(model, edge_splits, target_etypes, 'val', device)
        epoch_aucs = []
        epoch_aps = []

        for etype in target_etypes:
            if f"{etype}_AUC" in val_metrics and f"{etype}_AP" in val_metrics:
                auc_values = val_metrics[f"{etype}_AUC"]
                ap_values = val_metrics[f"{etype}_AP"]

                if auc_values and ap_values:
                    avg_auc = np.mean(auc_values)
                    avg_ap = np.mean(ap_values)
                    epoch_aucs.append(avg_auc)
                    epoch_aps.append(avg_ap)

                    print(f"  {etype} Val AUC: {avg_auc:.4f}, AP: {avg_ap:.4f}")

        if epoch_aucs:
            avg_val_auc = np.mean(epoch_aucs)
            print(f"  Average Val AUC: {avg_val_auc:.4f}")

        if epoch_aps:
            avg_ap_all = np.mean(epoch_aps)
            print(f"  Average Val AP: {avg_ap_all:.4f}")

        # Test set evaluation  
        print("\nTest Set Results:")
        test_metrics, _ = self.evaluate_on_split(model, edge_splits, target_etypes, 'test', device)
        epoch_aucs = []
        epoch_aps = []

        for etype in target_etypes:
            if f"{etype}_AUC" in test_metrics and f"{etype}_AP" in test_metrics:
                auc_values = test_metrics[f"{etype}_AUC"]
                ap_values = test_metrics[f"{etype}_AP"]

                if auc_values and ap_values:
                    avg_auc = np.mean(auc_values)
                    avg_ap = np.mean(ap_values)
                    epoch_aucs.append(avg_auc)
                    epoch_aps.append(avg_ap)

                    print(f"  {etype} Test AUC: {avg_auc:.4f}, AP: {avg_ap:.4f}")

        if epoch_aucs:
            avg_val_auc = np.mean(epoch_aucs)
            print(f"  Average Test AUC: {avg_val_auc:.4f}")

        if epoch_aps:
            avg_ap_all = np.mean(epoch_aps)
            print(f"  Average Test AP: {avg_ap_all:.4f}")