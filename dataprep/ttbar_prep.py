import h5py
import numpy as np
import sys
from tqdm import tqdm
import uproot

def main():

    file = uproot.open(sys.argv[1])['Delphes']

    jet_pt = file['jet_pt'].array(library='np')
    jet_eta = file['jet_eta'].array(library='np')
    jet_phi = file['jet_phi'].array(library='np')
    jet_btag = file['jet_bTag'].array(library='np')
    jet_m = file['jet_m'].array(library='np')
    jet_e = file['jet_e'].array(library='np')
    jet_matched = file['jet_truthmatch'].array(library='np')

    n_events = file.num_entries

    n_max_part = 20
    n_max_edges = n_max_part * (n_max_part - 1) // 2

    delphes_kin = np.zeros((n_events, n_max_part, 4))
    delphes_tag = np.zeros((n_events, n_max_part))

    aux_tag = np.zeros((n_events, n_max_part)) - 1

    edge_kin = np.zeros((n_events, n_max_edges, 3))
    adj_matrix = np.zeros((n_events, n_max_edges))

    top_idx = np.ones((n_events, 2, 3), dtype=int) * -1

    
    for i in tqdm(range(n_events), total=n_events, desc='Processing events'):

        n_nodes = len(jet_pt[i])
        n_edges = n_nodes * (n_nodes - 1) // 2

        delphes_kin[i, :n_nodes, 0] = jet_pt[i]
        delphes_kin[i, :n_nodes, 1] = jet_eta[i]
        delphes_kin[i, :n_nodes, 2] = jet_phi[i]
        delphes_kin[i, :n_nodes, 3] = jet_m[i]

        delphes_tag[i, :n_nodes] = jet_btag[i]

        edge_kin[i, :n_edges] = get_edge_kin(jet_pt[i], jet_eta[i], jet_phi[i], jet_e[i])

        adj_matrix[i, :n_edges] = get_adj_matrix(jet_matched[i])

        top_idx[i] = get_top_indices(jet_matched[i])

        aux_tag[i, :n_nodes] = get_auxtag(jet_matched[i])

    output_file = h5py.File(sys.argv[2], 'w')
    events_group = output_file.create_group('events')

    events_group.create_dataset('delphes_kin', data=delphes_kin)
    events_group.create_dataset('adj_matrix', data=adj_matrix)
    events_group.create_dataset('delphes_tag', data=delphes_tag)
    events_group.create_dataset('edges_kin', data=edge_kin)
    events_group.create_dataset('top_idx', data=top_idx)
    events_group.create_dataset('aux_tag', data=aux_tag)

    output_file.close()


def get_edge_kin(jet_pt, jet_eta, jet_phi, jet_e):
    # expand everything 
    row_idx, col_idx = np.triu_indices(len(jet_pt), k=1)
    jet_pt = np.tile(jet_pt, (len(jet_pt), 1))
    jet_eta = np.tile(jet_eta, (len(jet_eta), 1))
    jet_phi = np.tile(jet_phi, (len(jet_phi), 1))
    jet_e = np.tile(jet_e, (len(jet_e), 1))

    jpt = jet_pt[row_idx, col_idx]
    jpt_T = jet_pt[col_idx, row_idx]
    jeta = jet_eta[row_idx, col_idx]
    jeta_T = jet_eta[col_idx, row_idx]
    jphi = jet_phi[row_idx, col_idx]
    jphi_T = jet_phi[col_idx, row_idx]
    je = jet_e[row_idx, col_idx]
    je_T = jet_e[col_idx, row_idx]

    # calculate the edge kinematics
    # get delta R 
    delta_R = np.sqrt((jeta - jeta_T)**2 + (jphi - jphi_T)**2)
    # get inv mass
    inv_mass = inv_mass2(jpt, jpt_T, jeta, jeta_T, jphi, jphi_T, je, je_T)
    # get pt ratio
    pt_diff = np.abs(jpt - jpt_T) / (jpt + jpt_T)

    # concatenate everything 
    edge_kin = np.concatenate((delta_R[:, np.newaxis], inv_mass[:, np.newaxis], pt_diff[:, np.newaxis]), axis=1)
    return edge_kin


def inv_mass2(pt1, pt2, eta1, eta2, phi1, phi2, E1, E2):

    total_E = E1 + E2
    total_px = pt1 * np.cos(phi1) + pt2 * np.cos(phi2)
    total_py = pt1 * np.sin(phi1) + pt2 * np.sin(phi2)
    total_pz = pt1 * np.sinh(eta1) + pt2 * np.sinh(eta2)
    total_mass_sq = np.maximum(0, total_E**2 - (total_px**2 + total_py**2 + total_pz**2))
    total_mass = np.sqrt(total_mass_sq)
    return total_mass

def get_adj_matrix(match):
    n_nodes = match.shape[0]
    match_tile = np.tile(match, (n_nodes, 1))
    comb = np.concatenate([match_tile[:,:,np.newaxis], match_tile.T[:,:,np.newaxis]], axis=2)
    sorted = np.sort(comb, axis=2)
    adj = (np.all(sorted == np.array([2,3]), axis = -1) + np.all(sorted == np.array([5,6]), axis = -1)).astype(int)
    row, col = np.triu_indices(n_nodes, k=1)
    adj = adj[row, col]
    
    return adj

def get_top_indices(match):
    top1 = np.array([1, 2, 3])
    top2 = np.array([4, 5, 6])

    output = np.zeros((2, 3), dtype=int) - 1

    for i, top in enumerate([top1, top2]):
        if np.isin(top, match).all():
            arr = np.sort(np.array([np.where(match == t)[0][0] for t in top]))
            output[i] = arr

    return output

def get_auxtag(match):

    # 0 - fisr/isr, 1 - reco W, 2 - not reco W, 3 - reco t, 4 - not reco t
    aux_tag = np.zeros_like(match) - 1

    aux_tag[match == 0] = 0

    if np.isin([2,3], match).all():
        aux_tag[match == 2] = 1
        aux_tag[match == 3] = 1
        if 1 in match:
            aux_tag[match == 1] = 3
    else:
        aux_tag[match == 2] = 2
        aux_tag[match == 3] = 2
        if 1 in match:
            aux_tag[match == 1] = 4

    if np.isin([5,6], match).all():
        aux_tag[match == 5] = 1
        aux_tag[match == 6] = 1
        if 4 in match:
            aux_tag[match == 4] = 3
    else:
        aux_tag[match == 5] = 2
        aux_tag[match == 6] = 2
        if 4 in match:
            aux_tag[match == 4] = 4

    return aux_tag

        

if __name__ == "__main__":
    main()