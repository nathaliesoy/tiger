import torch 
import numpy as np 
import torch.nn.functional as F


class TigerLoss:
    def __init__(self):

        self.hyperedge_loss = torch.nn.CrossEntropyLoss(reduction='none')
        self.matching_loss = torch.nn.BCELoss(reduction='none')


    def stage1_edge_loss(self, edge_logits, adj_matrix, binary = True, edge_mask = None): # binary cross entropy loss for edge prediction
        
        adj_matrix = adj_matrix.flatten()
        edge_logits = edge_logits.flatten(0,1)

        if edge_mask is not None:
            edge_mask = edge_mask.flatten()
            edge_logits = edge_logits[edge_mask]
            adj_matrix = adj_matrix[edge_mask]

        # check for nans in logits
        if binary:
            edge_loss = F.binary_cross_entropy_with_logits(edge_logits, adj_matrix, reduction='none')
        else: # cross entropy loss for multi-class edge prediction
            edge_loss = F.cross_entropy(edge_logits, adj_matrix.long(), reduction='none')
            
        edge_loss_fin = 0
        for k in torch.unique(adj_matrix):
            edge_loss_k = edge_loss[adj_matrix == k].sum() / (adj_matrix == k).sum()
            edge_loss_fin += edge_loss_k
        return edge_loss_fin
    
    def stage2_edge_loss(self, edge_logits, meta_data_jetidx, edge_prob_mask, top_truth, node_mask = None):
        """
        edge_logits: shape (batch_size, num_nodes, top_k)
        meta_data_jetidx: shape (batch_size, top_k, 2)
        edge_prob_mask: shape (batch_size, top_k)
        top_truth: shape (batch_size, 2, 3)
        """
        num_nodes = edge_logits.size(1)
        batch_size = edge_logits.size(0)
        top_k = edge_logits.size(2)
        

        # combining tuples 
        nodes_jet_idx = torch.arange(num_nodes, device = edge_logits.device).repeat(batch_size, 1).unsqueeze(2) # shape (batch_size, num_nodes, 1)
        nodes_jet_idx_expanded = nodes_jet_idx.unsqueeze(2).expand(-1, -1, top_k, -1) # shape (batch_size, num_nodes, top_k, 1)
        meta_data_jetidx_expanded = meta_data_jetidx.unsqueeze(1).expand(-1, num_nodes, -1, -1) # shape (batch_size, num_nodes, top_k, 2)
        combined_jet_idx = torch.cat((nodes_jet_idx_expanded, meta_data_jetidx_expanded), dim=-1) # shape (batch_size, num_nodes, top_k, 3)
        # now sort them along last dimension
        combined_jet_idx = combined_jet_idx.sort(dim=-1).values
        
        combined_jet_idx = combined_jet_idx.unsqueeze(3) # shape (batch_size, num_nodes, top_k, 1,  3)
        top_truth_expanded = top_truth.unsqueeze(1).unsqueeze(1) # shape (batch_size, 1, 1, 2, 3)
        matches = (combined_jet_idx == top_truth_expanded).all(dim=-1)  # shape (batch_size, num_nodes, top_k, 2)
        target = matches.any(dim=-1).float()  # shape (batch_size, num_nodes, top_k)

        expanded_edge_mask = edge_prob_mask.unsqueeze(1).repeat(1, edge_logits.size(1), 1)

        target_flat = target.flatten()
        edge_logits_flat = edge_logits.flatten()
        expanded_edge_mask_flat = expanded_edge_mask.flatten()


        if node_mask is not None:
            node_mask = node_mask.unsqueeze(2).expand(-1, -1, edge_logits.size(2))
            node_mask_flat = node_mask.flatten()
            expanded_edge_mask_flat = expanded_edge_mask_flat * node_mask_flat

        target_flat = target_flat[expanded_edge_mask_flat]
        edge_logits_flat = edge_logits_flat[expanded_edge_mask_flat]

        bce_loss = F.binary_cross_entropy_with_logits(edge_logits_flat, target_flat, reduction='none')

        edge_loss_fin = 0
        for k in torch.unique(target_flat):
            edge_loss_k = bce_loss[target_flat == k].sum() / (target_flat == k).sum()
            edge_loss_fin += edge_loss_k

        return edge_loss_fin, target
         

    def aux_loss(self, aux_pred, aux_target, node_mask=None):
        """
        aux_pred: shape (batch_size, aux_dim)
        aux_target: shape (batch_size)
        """
        # flatten the tensors
        aux_pred = aux_pred.flatten(0, 1)
        aux_target = aux_target.flatten(0, 1).long()
        if node_mask is not None:
            node_mask = node_mask.flatten(0, 1)
            aux_pred = aux_pred[node_mask]
            aux_target = aux_target[node_mask]

        aux_loss = F.cross_entropy(aux_pred, aux_target.long(), reduction='none')
        aux_loss_fin = aux_loss.mean()

        return aux_loss_fin

    
    def aux_loss_fin(self, aux_pred, aux_target):
        """
        aux_pred: shape (batch_size, aux_dim)
        aux_target: shape (batch_size)
        """
        # flatten the tensors
        aux_pred = aux_pred
        aux_target = aux_target.long()
        
        aux_loss = F.cross_entropy(aux_pred, aux_target.long(), reduction='none')
        
        # return aux_loss.mean()
        aux_loss_fin = 0
        for k in torch.unique(aux_target):
            aux_loss_k = aux_loss[aux_target == k].sum() / (aux_target == k).sum()
            aux_loss_fin += aux_loss_k

        return aux_loss_fin
    
    def sb_class_loss(self, sb_pred, sb_target):
        """
        sb_pred: shape (batch_size)
        sb_target: shape (batch_size)
        """
        sb_loss = F.binary_cross_entropy_with_logits(sb_pred, sb_target.float(), reduction='none')

        return sb_loss.mean()

       



        

