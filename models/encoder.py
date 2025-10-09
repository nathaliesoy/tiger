import torch 
import torch.nn as nn
from models.diffusion_transformer import DiTEncoder

def one_hot_encoding(pids, possible_pdgids):

        pdg_id_to_idx = {pdgid: idx for idx, pdgid in enumerate(possible_pdgids)}

        mapping_tensor = torch.full((max(possible_pdgids) + 1,), -1, dtype=torch.long).to(pids.device)

        for pdgid, idx in pdg_id_to_idx.items():
            mapping_tensor[pdgid] = idx

        indices = mapping_tensor[pids.long()]

        one_hot_vector = torch.zeros(*pids.shape, len(possible_pdgids), dtype=torch.float, device=pids.device)
        one_hot_vector.scatter_(-1, indices.unsqueeze(-1), 1)

        return one_hot_vector

class EncoderModel(nn.Module):
    def __init__(self, config):
        super().__init__()

        self.config = config

        self.use_sampler = config.get('use_sampler', True)

        self.input_encoding = build_layers(self.config['encoder']['input_dim'], self.config['hid_dim'], self.config['encoder']['init_hidden_dim'], dropout=self.config['dropout'])

        if config.get('split_input_classes', False):
            self.lep_encoding = build_layers(self.config['encoder']['input_dim'], self.config['hid_dim'], self.config['encoder']['init_hidden_dim'], dropout=self.config['dropout'])
            self.met_encoding = build_layers(3, self.config['hid_dim'], self.config['encoder']['init_hidden_dim'], dropout=self.config['dropout'])
        
        self.trf_config = self.config['encoder']['transformer']

        self.transformer = DiTEncoder(self.config['hid_dim'], 
                                        self.trf_config['n_trf'], 
                                        mha_config = self.trf_config['mha_config'], 
                                        dense_config = self.trf_config['dense_config'], 
                                        context_dim=self.config['hid_dim'] + 1, 
                                        out_dim=self.config['hid_dim'])



    def forward(self, data):

        if self.config.get('split_input_classes', False):
            encoded_pid = one_hot_encoding(data['delphes_tag'][:, 1:-1], self.config['possible_pdgids'])
            encoded_pid_lep = one_hot_encoding(data['delphes_tag'][:, 0], self.config['lep_pdgids'])

            combined_jet_input = torch.cat((data['delphes_kin'][:, 1:-1, :], encoded_pid), 2)
            combined_lep_input = torch.cat((data['delphes_kin'][:, 0, :].unsqueeze(1), encoded_pid_lep.unsqueeze(1)), 2)
            lep_input = self.lep_encoding(combined_lep_input)
            jet_input = self.input_encoding(combined_jet_input)
            met_input = self.met_encoding(data['delphes_kin'][:, -1, [0,2,3]]).unsqueeze(1)
            combined_input = torch.cat((lep_input, jet_input, met_input), 1)

        else:
            encoded_pid = one_hot_encoding(data['delphes_tag'], self.config['possible_pdgids'])
            combined_input = torch.cat((data['delphes_kin'], encoded_pid), 2)
            combined_input = self.input_encoding(combined_input)
            
        data['hidden_rep_0'] = combined_input
        data['hidden_rep'] = data['hidden_rep_0']
      
        n_scale = ((data['n_objects'] - self.config['n_min']) / (self.config['n_max'] - self.config['n_min'])).unsqueeze(-1)
        data['n_obj_scale'] = n_scale

        if self.use_sampler is False:
            mask = torch.arange(combined_input.shape[1], device = combined_input.device).unsqueeze(0) < data['n_objects'].unsqueeze(1)
            data['node_mask'] = mask
            context = torch.cat([torch.sum(data['hidden_rep_0'] * mask.unsqueeze(-1), 1) / data['n_objects'].unsqueeze(-1), n_scale], dim = -1)
        else:
            context = torch.cat([torch.mean(data['hidden_rep_0'], 1), n_scale], dim = -1)

        data['hidden_rep'] = self.transformer(
            q = data['hidden_rep'],
            q_mask = (~mask if self.use_sampler is False else None),
            context = context
        )

        data['hidden_rep'] = data['hidden_rep'] * mask.unsqueeze(-1) if self.use_sampler is False else data['hidden_rep']

        return data



def build_layers(inputsize,outputsize,features, add_batch_norm=False,add_activation=None, dropout= None):
    layers = []
    layers.append(nn.Linear(inputsize,features[0]))
    layers.append(nn.LeakyReLU())
    
    for hidden_i in range(1,len(features)):
        if add_batch_norm:
            layers.append(nn.BatchNorm1d(features[hidden_i-1]))
        
        layers.append(nn.Linear(features[hidden_i-1],features[hidden_i]))
        layers.append(nn.LeakyReLU())
    if dropout is not None:
        layers.append(nn.Dropout(dropout))

    layers.append(nn.Linear(features[-1],outputsize))
    
    if add_activation!=None:
        layers.append(add_activation)
    return nn.Sequential(*layers)