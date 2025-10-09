import torch 
import torch.nn as nn
import torch.nn.functional as F
from models.encoder import EncoderModel, build_layers
from models.diffusion_transformer import DiTEncoder


class TigerModel(nn.Module):
    def __init__(self, config):
        super().__init__()

        def initialize_weights(layer):
            if isinstance(layer, nn.Linear):
                nn.init.xavier_uniform_(layer.weight)
                if layer.bias is not None:
                    nn.init.zeros_(layer.bias)

        self.config = config
        self.use_sampler = config.get('use_sampler', True)

        self.use_edge_features = self.config['use_edge_features']

        self.encoder = EncoderModel(self.config) # this should encode the data, create updated hidden reps

        if self.use_edge_features:
            self.edge_net = build_layers(self.config['hid_dim'] + 3, self.config['hid_dim'], self.config['edge_net1_hid_dim'], dropout=self.config['dropout'])
        else:
            self.edge_net = build_layers(self.config['hid_dim'], self.config['hid_dim'], self.config['edge_net1_hid_dim'], dropout=self.config['dropout'])

        if self.config['binary_class']:
            self.edge_classifier = build_layers(self.config['hid_dim'], 1, self.config['edge_class1_hid_dim'], dropout=self.config['dropout'])
        else:
            self.edge_classifier = build_layers(self.config['hid_dim'], 3, self.config['edge_class1_hid_dim'], dropout=self.config['dropout'])

        self.do_aux_task = False
        self.do_aux_task_fin = False
        self.do_sb_class = False

        if isinstance(config, dict) and config.get('aux_task', False):
            self.do_aux_task = True
            
            self.aux_net = build_layers(self.config['hid_dim'], 
                                        self.config['aux_classes'],
                                        self.config['aux_net_hid_dim'])
            
            self.aux_net.apply(initialize_weights)

        

        if not self.config['stage1_only']:

            if isinstance(config, dict) and config.get('aux_task_fin', False):
                self.do_aux_task_fin = True
                self.aux_net_fin = build_layers(2 * self.config['hid_dim'] + 2, 
                                            self.config['aux_classes_fin'],
                                            self.config['aux_net_hid_dim_fin'])#, dropout = 0.2)
                self.aux_net_fin.apply(initialize_weights)

            self.top_k = self.config['top_k'] 
            self.prob_threshold = self.config['edge_prob_threshold']

            self.node_projector = build_layers(self.config['hid_dim'], self.config['hid_dim'], self.config['node_proj_hid_dim'], dropout=self.config['dropout'])
            self.meta_node_projector = build_layers(self.config['hid_dim'] + 1, self.config['hid_dim'], self.config['node_proj_hid_dim'], dropout=self.config['dropout'])

            self.trf_config = self.config['stage2_transformer']
            self.stage2_transformer = DiTEncoder(self.config['hid_dim'], 
                                        self.trf_config['n_trf'], 
                                        mha_config = self.trf_config['mha_config'], 
                                        dense_config = self.trf_config['dense_config'], 
                                        context_dim=self.config['hid_dim'] + 2, 
                                        out_dim=self.config['hid_dim'])

            
            self.stage2_edge_net = build_layers(self.config['hid_dim'], self.config['hid_dim'], self.config['edge_net2_hid_dim'], dropout=self.config['dropout'])

            self.stage2_edge_classifier = build_layers(self.config['hid_dim'], 1, self.config['edge_class2_hid_dim'], dropout=self.config['dropout'])
            
            if config.get('sb_class', False):
                self.do_sb_class = True
                self.sb_classifier = build_layers(2 * self.config['hid_dim'] + 2, 1, self.config['sb_class_net_hid_dim'], dropout=self.config['dropout'])
                self.sb_classifier.apply(initialize_weights)

            self.node_projector.apply(initialize_weights)
            self.meta_node_projector.apply(initialize_weights)
            self.stage2_transformer.apply(initialize_weights)
            self.stage2_edge_net.apply(initialize_weights)
            self.stage2_edge_classifier.apply(initialize_weights)

        self.edge_net.apply(initialize_weights)
        self.edge_classifier.apply(initialize_weights)
        self.encoder.apply(initialize_weights)

    def forward(self, data):
        data = self.encoder(data)
        num_nodes = data['hidden_rep'].shape[1]
        bs = data['hidden_rep'].shape[0]

        #check for nans in hidden_rep
        if torch.isnan(data['hidden_rep']).any():
            raise ValueError("NaN values found in hidden_rep")

        if self.do_aux_task:
            aux_pred = self.aux_net(data['hidden_rep'])
            data['aux_pred'] = aux_pred

        hidden_rep1 = data['hidden_rep'].unsqueeze(2).repeat(1, 1, num_nodes, 1) # shape: (batch_size, num_nodes, num_nodes, hid_dim)
        hidden_rep2 = hidden_rep1.transpose(1, 2) 

        hidden_rep_sum = hidden_rep1 + hidden_rep2
        
        row_idx, col_idx = torch.triu_indices(num_nodes, num_nodes, offset=1, device=data['hidden_rep'].device)
        
        pairwise_hidden_rep = hidden_rep_sum[:, row_idx, col_idx, :]

        if self.use_edge_features:
            pairwise_hidden_rep = torch.cat((pairwise_hidden_rep, data['edges_kin']), dim=-1) # shape: (batch_size, N*(N-1)//2 , 2 * hid_dim + 3)
        
        edge_rep = self.edge_net(pairwise_hidden_rep) # shape: (batch_size, num_nodes, num_nodes, hid_dim)
        edge_pred = self.edge_classifier(edge_rep) # shape: (batch_size, num_nodes, num_nodes)
        
        if self.config['binary_class']:
            edge_pred = edge_pred.squeeze(-1)
        data['edge_pred'] = edge_pred
        data['edge_rep'] = edge_rep
        
        if not self.config['stage1_only']:
            data = self.forward_stage2_eval(data, row_idx, col_idx)

        return data

    
    def forward_stage2_eval(self, data, row_idx, col_idx):
            
        edge_pred = data['edge_pred']
        edge_rep = data['edge_rep']
        bs = edge_pred.shape[0]
        num_nodes = data['hidden_rep'].shape[1]
        if not self.use_sampler:
            edge_mask = data['edge_mask']  
            edge_pred[edge_mask == False] = -1e9  # set invalid edges to a very low value

        if self.config['binary_class']:
            prob_sorted, idx_triu = torch.sort(edge_pred, descending=True)
            converted_prob = torch.sigmoid(prob_sorted[:, :self.top_k])
        else:
            prob_softmax = F.softmax(edge_pred, dim=-1)
            prob_W = prob_softmax[:,:,1]
            prob_sorted, idx_triu = torch.sort(prob_W, descending=True)
            converted_prob = prob_sorted[:, :self.top_k]

        prob_mask = converted_prob > self.prob_threshold

        meta_data_jetidx = torch.cat((row_idx[idx_triu[:, :self.top_k]].unsqueeze(2), col_idx[idx_triu[:, :self.top_k]].unsqueeze(2)), dim = 2 )
        meta_data_jetidx[~prob_mask,:] = -1
        data['meta_data_jetidx'] = meta_data_jetidx
        data['edge_prob_mask'] = prob_mask
        meta_nodes = edge_rep[torch.arange(bs).unsqueeze(1), idx_triu[:, :self.top_k]]

        node_proj = self.node_projector(data['hidden_rep'])


        meta_node_proj = self.meta_node_projector(torch.cat((meta_nodes, prob_sorted[:, :self.top_k].unsqueeze(2)), dim = 2))
        
        # check for nans in node_proj and meta_node_proj
        if torch.isnan(node_proj).any():
            raise ValueError("NaN values found in node_proj")
        if torch.isnan(meta_node_proj).any():
            raise ValueError("NaN values found in meta_node_proj")

        merged_nodes = torch.cat((node_proj, meta_node_proj), dim=1) # shape: (batch_size, num_nodes + top_k, hid_dim)

        if self.use_sampler:
            q_mask = torch.cat((torch.ones(bs, num_nodes, dtype=torch.bool, device=edge_pred.device),
                                prob_mask), dim=1)  # shape: (batch_size, num_nodes + top_k)
        else:
            q_mask = torch.cat((data['node_mask'], prob_mask), dim=1)
        

        n_scale = ((data['n_objects'] - self.config['n_min']) / (self.config['n_max'] - self.config['n_min'])).unsqueeze(-1)

        n_scale_meta = (prob_mask.sum(dim=1) / self.top_k).unsqueeze(-1)
        
        context = torch.cat([torch.sum( merged_nodes * q_mask.unsqueeze(-1).float(), 1) / torch.sum(q_mask, dim=1).unsqueeze(-1),
                            n_scale, n_scale_meta], dim = -1)
        
        all_updated_nodes = self.stage2_transformer(
            q = merged_nodes,# * q_mask.unsqueeze(-1).float(),
            q_mask = ~q_mask,
            context = context
        )

        all_updated_nodes = all_updated_nodes * q_mask.unsqueeze(-1).float()
        
        # split and reshape nodes into (bs, num_nodes, top_k, hid_dim) and (bs, top_k, num_nodes, hid_dim)
        updated_nodes = all_updated_nodes[:, :num_nodes, :].unsqueeze(2).repeat(1, 1, self.top_k, 1) # shape: (batch_size, num_nodes, top_k, hid_dim)
        updated_meta_nodes = all_updated_nodes[:, num_nodes:, :].unsqueeze(1).repeat(1, num_nodes, 1, 1) # shape: (batch_size, num_nodes, top_k, hid_dim)
        
        pairwise_node_sum = updated_nodes + updated_meta_nodes # shape: (batch_size, num_nodes, top_k, hid_dim)
        edge_rep2 = self.stage2_edge_net(pairwise_node_sum) # shape: (batch_size, num_nodes, top_k, hid_dim)
        edge_pred2 = self.stage2_edge_classifier(edge_rep2)

    
        data['edge_pred2'] = edge_pred2.squeeze(-1)

        combined_output = torch.cat((all_updated_nodes[:, :num_nodes, :].sum(dim=1) / num_nodes,
                                        all_updated_nodes[:, num_nodes:, :].sum(dim=1) / (prob_mask + 1e-8).sum(dim=1).unsqueeze(-1),
                                        n_scale, n_scale_meta), dim=-1)
            

        if self.do_aux_task_fin:

            aux_pred_fin = self.aux_net_fin(combined_output)
            data['aux_pred_fin'] = aux_pred_fin

        if self.do_sb_class:
            sb_pred = self.sb_classifier(combined_output)
            data['sb_pred'] = sb_pred.squeeze(1)                    
        return data
    