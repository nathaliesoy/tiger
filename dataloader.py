import torch 
import h5py
from tqdm import tqdm
import numpy as np
import uproot 
import awkward as ak

from torch.utils.data import Dataset, Sampler


class tiger_dataset(Dataset):

    def __init__(self, path, config, n_events = None, event_start_idx = 0):

        self.path = path
        self.n_events = n_events
        self.config = config

        tth_data = self.config.get('tth_data', False)

        data_keys_to_read = ['delphes_kin', 'delphes_tag', 'adj_matrix', 'edges_kin', 'top_idx']
        # check if an attribute is in the config
        if config.get('aux_task', False):
            data_keys_to_read.append('aux_tag')

        if config.get('sb_class', False):
            data_keys_to_read.append('signal')
        
        self.data_dict = {}

        print(f'Loading file ... ')
        input = h5py.File(self.path, 'r')['events']
        # input = uproot.open(self.path)['events']

        for key in data_keys_to_read:
            self.data_dict[key] = input[key][event_start_idx : self.n_events + event_start_idx]

        self.data_dict['n_objects'] = np.array([np.sum(self.data_dict['delphes_kin'][i, :, 0] >0) for i in range(len(self.data_dict['delphes_kin']))])
        self.data_dict['index'] = np.arange(self.n_events) + event_start_idx

        # resacle pT and mass
        self.data_dict['delphes_kin'][:, :, 0] = (np.log(self.data_dict['delphes_kin'][:, :, 0] + 1) - self.config['transform']['pt']['mean']) / self.config['transform']['pt']['sigma']
        
        if config['dataset'] == 'hyper':
            self.data_dict['delphes_kin'][:, :, 3] = np.log(self.data_dict['delphes_kin'][:, :, 3] + 1) 
        elif config['dataset'] == 'spanet':
            self.data_dict['delphes_kin'][:, :, 4] = np.log(self.data_dict['delphes_kin'][:, :, 4] + 1) 
        else:
            raise ValueError(f"Unknown dataset: {config['dataset']}")

        self.data_dict['edges_kin'][:, :, 1] = np.log(self.data_dict['edges_kin'][:, :, 1] + 1) 

        # sort by last dimension of top_idx
        self.data_dict['top_idx'] = np.sort(self.data_dict['top_idx'], axis=-1)

        if config.get('aux_task_fin', False):
            n_tops = np.any(self.data_dict['top_idx'] != -1, axis = 2).sum(axis = 1) 

            if tth_data:
                n_higgs = np.any(self.data_dict['adj_matrix'] == 2, axis =1)#.sum(axis = 1)
                self.data_dict['aux_tag_fin'] = 3 * n_higgs + n_tops
            else:
                self.data_dict['aux_tag_fin'] = n_tops


        self.n_events = len(self.data_dict['delphes_kin'])

    def __len__(self):
        return self.n_events
    
    def __getitem__(self, idx):
        data = {key: torch.tensor(self.data_dict[key][idx]) for key in self.data_dict.keys()}
        return data
    


def collate_fn(samples, sampler = True):

    bs = len(samples)
    if sampler:
        n_obs = samples[0]['n_objects'].item()
    else:
        n_obs = max([sample['n_objects'].item() for sample in samples])
    n_edges = (n_obs * (n_obs - 1)) // 2

    batched_dict = {
        
        'delphes_kin': torch.zeros(bs, n_obs, samples[0]['delphes_kin'].shape[1]),
        'delphes_tag': torch.zeros(bs, n_obs),
        'adj_matrix': torch.zeros(bs, n_edges, dtype=torch.double) - 1,
        'edges_kin': torch.zeros(bs, n_edges, samples[0]['edges_kin'].shape[1]),
        'top_idx': torch.zeros(bs, 2, 3),
        }
    
    if 'aux_tag' in samples[0]:
        batched_dict['aux_tag'] = torch.zeros(bs, n_obs)

    if 'aux_tag_fin' in samples[0]:
        batched_dict['aux_tag_fin'] = torch.zeros(bs)

    if 'signal' in samples[0]:
        batched_dict['signal'] = torch.zeros(bs)

    if sampler:
        for i, sample in enumerate(samples):
            for key in batched_dict.keys():
                if batched_dict[key].ndim == 2:
                    batched_dict[key][i] = sample[key][:batched_dict[key].shape[1]]
                elif batched_dict[key].ndim == 3:
                    batched_dict[key][i] = sample[key][:batched_dict[key].shape[1], :batched_dict[key].shape[2]]
                elif batched_dict[key].ndim == 1:
                    batched_dict[key][i] = sample[key]

    else:
        batched_dict['edge_mask'] = torch.zeros(bs, n_edges, dtype=torch.bool)
        for i, sample in enumerate(samples):
            n = sample['n_objects'].item()
            e = n*(n - 1) // 2
            sample_row, sample_col = torch.triu_indices(n, n, offset=1)
            sample_edge_indices = []
            for r, c in zip(sample_row.tolist(), sample_col.tolist()):
                global_idx = ((2 * n_obs - r - 1) * r) // 2 + (c - r - 1)
                sample_edge_indices.append(global_idx)

            sample_edge_indices = torch.tensor(sample_edge_indices, dtype=torch.int)
            
            batched_dict['adj_matrix'][i, sample_edge_indices] = sample['adj_matrix'][:e]
            batched_dict['edges_kin'][i, sample_edge_indices] = sample['edges_kin'][:e].to(torch.float)
            batched_dict['edge_mask'][i, sample_edge_indices] = True

            for key in batched_dict.keys(): 
                if key not in ['adj_matrix', 'edges_kin', 'edge_mask']:
                    if batched_dict[key].ndim == 2:
                        batched_dict[key][i, :n] = sample[key][:n]
                    elif batched_dict[key].ndim == 3:
                        batched_dict[key][i, :n, :] = sample[key][:n, :]
                    elif batched_dict[key].ndim == 1:
                        batched_dict[key][i] = sample[key]

           
    batched_dict['n_objects'] = torch.tensor([sample['n_objects'] for sample in samples])
    batched_dict['index'] = torch.tensor([sample['index'] for sample in samples])

    return batched_dict



class TigerSampler(Sampler):
    def __init__(self, n_nodes_array, batch_size, shuffle = True):
        """
        Args:
            n_nodes_array: array of the number of nodes (tracks + topos)
            batch_size: batch size
        """
        super().__init__()
        self.dataset_size = len(n_nodes_array)
        self.batch_size = batch_size

        self.index_to_batch = {}
        running_idx = -1
        self.shuffle = shuffle

        for n_nodes_i in np.unique(n_nodes_array):

            n_nodes_idxs = np.where(n_nodes_array == n_nodes_i)[0]
            
            indices = np.arange(0, len(n_nodes_idxs), self.batch_size)
            n_nodes_idxs = [n_nodes_idxs[i: i + self.batch_size] for i in indices]

            for batch in n_nodes_idxs:
                running_idx += 1
                self.index_to_batch[running_idx] = batch

        self.n_batches = running_idx + 1
        print(f'Number of batches: {self.n_batches}')

    def __len__(self):
        return self.n_batches

    def __iter__(self):
        if self.shuffle:
            batch_order = np.random.permutation(np.arange(self.n_batches))
        else:
            batch_order = np.arange(self.n_batches)
        for i in batch_order:
            yield self.index_to_batch[i]
            
            


