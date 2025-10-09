import numpy as np
import h5py
import sys
from tqdm import tqdm

def main():

    f = h5py.File(sys.argv[1], 'r')

    get_signal = False
    if 'background' in sys.argv[1]:
        get_signal = True

    mask = f['INPUTS']['Momenta']['MASK'][:]

    pt = f['INPUTS']['Momenta']['pt'][:]
    eta = f['INPUTS']['Momenta']['eta'][:]
    cos_phi = f['INPUTS']['Momenta']['cos_phi'][:]
    sin_phi = f['INPUTS']['Momenta']['sin_phi'][:]
    m = f['INPUTS']['Momenta']['mass'][:]

    btag = f['INPUTS']['Momenta']['btag'][:]
    etag = f['INPUTS']['Momenta']['etag'][:]
    utag = f['INPUTS']['Momenta']['utag'][:]

    met = f['INPUTS']['Met']['met'][:]
    met_cos_phi = f['INPUTS']['Met']['cos_phi'][:]
    met_sin_phi = f['INPUTS']['Met']['sin_phi'][:]

    t_h_b = f['TARGETS']['ht']['b'][:]
    t_h_q1 = f['TARGETS']['ht']['q1'][:]
    t_h_q2 = f['TARGETS']['ht']['q2'][:]
    t_l_b = f['TARGETS']['lt']['b'][:]
    t_l_l = f['TARGETS']['lt']['l'][:]
    h_b1 = f['TARGETS']['h']['b1'][:]
    h_b2 = f['TARGETS']['h']['b2'][:]

    ## FOR TTH-TTBB
    if get_signal:
        signal = f['CLASSIFICATIONS']['EVENT']['signal'][:]
        ttbb_thb = f['TARGETS']['ttbb_ht']['b'][:]
        ttbb_thq1 = f['TARGETS']['ttbb_ht']['q1'][:]
        ttbb_thq2 = f['TARGETS']['ttbb_ht']['q2'][:]
        ttbb_tlb = f['TARGETS']['ttbb_lt']['b'][:]
        ttbb_tll = f['TARGETS']['ttbb_lt']['l'][:]

        t_h_b = np.where(signal ==1, t_h_b, ttbb_thb)
        t_h_q1 = np.where(signal ==1, t_h_q1, ttbb_thq1)
        t_h_q2 = np.where(signal ==1, t_h_q2, ttbb_thq2)
        t_l_b = np.where(signal ==1, t_l_b, ttbb_tlb)
        t_l_l = np.where(signal ==1, t_l_l, ttbb_tll)

    ## END HERE 

    jet_match = np.vstack([h_b1, h_b2, t_h_b, t_h_q1, t_h_q2, t_l_b, t_l_l]).T


    n_events = mask.shape[0]

    n_max_part = 20
    n_max_edges = n_max_part * (n_max_part - 1) // 2

    delphes_kin = np.zeros((n_events, n_max_part, 5)) # pt, eta, cos_phi, sin_phi, m
    delphes_tag = np.zeros((n_events, n_max_part))

    edge_kin = np.zeros((n_events, n_max_edges, 3))
    adj_matrix = np.zeros((n_events, n_max_edges))

    top_idx = np.ones((n_events, 2, 3), dtype=int) * -1

    aux_tag = np.zeros((n_events, n_max_part)) - 1


    for idx in tqdm(range(n_events), total=n_events, desc='Processing events'):
    
        n_nodes = mask[idx].sum() + 1
        n_edges = n_nodes * (n_nodes - 1) // 2

        delphes_kin[idx, :n_nodes - 1, 0] = pt[idx][mask[idx]]
        delphes_kin[idx, :n_nodes - 1, 1] = eta[idx][mask[idx]]
        delphes_kin[idx, :n_nodes - 1, 2] = cos_phi[idx][mask[idx]]
        delphes_kin[idx, :n_nodes - 1, 3] = sin_phi[idx][mask[idx]]
        delphes_kin[idx, :n_nodes - 1, 4] = m[idx][mask[idx]]


        delphes_kin[idx, n_nodes - 1, 0] = met[idx]
        delphes_kin[idx, n_nodes - 1, 2] = met_cos_phi[idx]
        delphes_kin[idx, n_nodes - 1, 3] = met_sin_phi[idx]
        delphes_tag[idx, n_nodes - 1] = 99

        delphes_tag[idx, np.where(btag[idx] == 1)[0]] = 1
        delphes_tag[idx, np.where(etag[idx] == 1)[0]] = 11
        delphes_tag[idx, np.where(utag[idx] == 1)[0]] = 13

        edge_kin[idx, :n_edges] = get_edge_kin(delphes_kin[idx,:n_nodes,0],
                                                delphes_kin[idx,:n_nodes,1],
                                                delphes_kin[idx,:n_nodes,2],
                                                delphes_kin[idx,:n_nodes,3],
                                                delphes_kin[idx,:n_nodes,4])

        adj_matrix[idx, :n_edges] = get_adj_matrix(jet_match[idx], n_nodes)

        top_idx[idx] = get_top_indices(jet_match[idx], n_nodes)
        aux_tag[idx, :n_nodes] = get_aux_tag(jet_match[idx], n_nodes)

    output_file = h5py.File(sys.argv[2], 'w')
    events_group = output_file.create_group('events')

    events_group.create_dataset('delphes_kin', data=delphes_kin)
    events_group.create_dataset('adj_matrix', data=adj_matrix)
    events_group.create_dataset('delphes_tag', data=delphes_tag)
    events_group.create_dataset('edges_kin', data=edge_kin)
    events_group.create_dataset('top_idx', data=top_idx)
    events_group.create_dataset('aux_tag', data=aux_tag)
    if get_signal:
        events_group.create_dataset('signal', data=signal)

    output_file.close()

def get_edge_kin(jet_pt, jet_eta, jet_cosphi, jet_sinphi, jet_m):
    # expand everything 
    row_idx, col_idx = np.triu_indices(len(jet_pt), k=1)
    jet_pt = np.tile(jet_pt, (len(jet_pt), 1))
    jet_eta = np.tile(jet_eta, (len(jet_eta), 1))
    jet_sinphi = np.tile(jet_sinphi, (len(jet_sinphi), 1))
    jet_cosphi = np.tile(jet_cosphi, (len(jet_cosphi), 1))
    jet_m = np.tile(jet_m, (len(jet_m), 1))

    jpt = jet_pt[row_idx, col_idx]
    jpt_T = jet_pt[col_idx, row_idx]
    jeta = jet_eta[row_idx, col_idx]
    jeta_T = jet_eta[col_idx, row_idx]
    jsinphi = jet_sinphi[row_idx, col_idx]
    jsinphi_T = jet_sinphi[col_idx, row_idx]
    jcosphi = jet_cosphi[row_idx, col_idx]
    jcosphi_T = jet_cosphi[col_idx, row_idx]
    jm = jet_m[row_idx, col_idx]
    jm_T = jet_m[col_idx, row_idx]

    jphi = np.arctan2(jsinphi, jcosphi)
    jphi_T = np.arctan2(jsinphi_T, jcosphi_T)

    # calculate the edge kinematics
    # get delta R 
    delta_R = np.sqrt((jeta - jeta_T)**2 + (jphi - jphi_T)**2)
    # get inv mass
    inv_mass = inv_mass2(jpt, jpt_T, jeta, jeta_T, jsinphi, jsinphi_T, jcosphi, jcosphi_T, jm, jm_T)
    # get pt ratio
    epsilon = 1e-8  # Small value to prevent division by zero
    pt_diff = np.abs(jpt - jpt_T) / (jpt + jpt_T + epsilon)

    # concatenate everything 
    edge_kin = np.concatenate((delta_R[:, np.newaxis], inv_mass[:, np.newaxis], pt_diff[:, np.newaxis]), axis=1)
    return edge_kin

def inv_mass2(pt1, pt2, eta1, eta2, sinphi1, sinphi2, cosphi1, cosphi2, m1, m2):

    pz1 = pt1 * np.sinh(eta1)
    pz2 = pt2 * np.sinh(eta2)
    E1 = np.sqrt(pt1**2 + pz1**2 + m1**2)
    E2 = np.sqrt(pt2**2 + pz2**2 + m2**2)

    total_E = E1 + E2
    total_px = pt1 * cosphi1 + pt2 * cosphi2
    total_py = pt1 * sinphi1 + pt2 * sinphi2
    total_pz = pz1 + pz2
    total_mass_sq = np.maximum(0, total_E**2 - (total_px**2 + total_py**2 + total_pz**2))
    total_mass = np.sqrt(total_mass_sq)
    return total_mass

def get_adj_matrix(match, n_nodes):
    # match is t_h_b, t_h_q1, t_h_q2, t_l_b, t_l_l
    adj = np.zeros((n_nodes, n_nodes))

    if (match[0] > -1) and (match[1] > -1): # higgs
        adj[match[0], match[1]] = 2

    if (match[3] > -1) and (match[4] > -1): # hadronic W
        adj[match[3], match[4]] = 1
    if (match[6] > -1): # match lepton with missing ET
        adj[match[6], n_nodes - 1 ] = 1
        # leptonic W gets different label
        # adj[match[4], n_nodes - 1 ] = 2
    row, col = np.triu_indices(n_nodes, k=1)
    adj = adj[row, col]
    
    return adj

def get_top_indices(match, n_nodes):
    output = np.zeros((2, 3), dtype=int) - 1

    if -1 in match[2:5]:  # hadronic top quark
        output[0] = [-1, -1, -1]  # no hadronic top quark
    else:
        output[0] = match[2:5]  

    if -1 in match[5:]:  # leptonic top quark
        output[1] = [-1, -1, -1]
    else:
        output[1,:2] = match[5:]
        output[1, 2] = n_nodes - 1 #match to missing ET 

    return output

def get_aux_tag(match, n_nodes):

    aux_tag = np.zeros(n_nodes)
    aux_tag[n_nodes - 1] = 8  # missing ET

    # check for W & had top 
    if (match[3:5] > -1).all():
        aux_tag[match[3:5]] = 1
        if match[2] > -1:
            aux_tag[match[2]] = 3
    else:
        aux_tag[match[3:5][match[3:5] > -1]] = 2
        if match[2] > -1:
            aux_tag[match[2]] = 4

    # b from leptonic top
    if match[5] > -1:
        aux_tag[match[5]] = 3

    # check for higgs 
    if (match[0:2] > -1).all():
        aux_tag[match[0:2]] = 5
    else:
        aux_tag[match[0:2][match[0:2] > -1]] = 6

    if match[6] > -1:
        aux_tag[match[6]] = 7

    return aux_tag


def get_inc_matrix(jet_match, mask):

    nj = np.sum(mask)
    nj_match = np.sum(jet_match > -1)

    edge_lab = []
    if np.any(jet_match[:2]) > -1:
        edge_lab.append(5)
    if np.any(jet_match[2:5]) > -1:
        edge_lab.append(1)
    if np.any(jet_match[5:]) > -1:
        edge_lab.append(2)
    if np.any(jet_match[3:5]) > -1:
        edge_lab.append(3)
    if jet_match[6] > -1:
        edge_lab.append(4)

    for i in range(nj - nj_match):
        edge_lab.append(0)

    edge_lab = np.array(edge_lab)

    ne = len(edge_lab)

    inc_init = np.zeros((nj+1, ne))

    for i in range(nj):
        idx = np.where(jet_match == i)[0]
        if idx.size == 0:
            continue
        if idx in [2]:
            inc_init[i][np.where(edge_lab == 1)[0]] = 1
        elif idx in [5]:
            inc_init[i][np.where(edge_lab == 2)[0]] = 1
        elif idx in [3, 4]:
            inc_init[i][np.where(edge_lab == 3)[0]] = 1
            inc_init[i][np.where(edge_lab == 1)[0]] = 1
        elif idx in [6]:
            inc_init[i][np.where(edge_lab == 4)[0]] = 1
            inc_init[i][np.where(edge_lab == 2)[0]] = 1
        elif idx in [0, 1]:
            inc_init[i][np.where(edge_lab == 5)[0]] = 1

    idx_unmatched = np.arange(nj)[~np.isin(np.arange(nj), jet_match)]
    idx_edge_unmatch = np.where(edge_lab == 0)[0]
    for i in range(len(idx_unmatched)):
        inc_init[idx_unmatched[i]][idx_edge_unmatch[i]] = 1

    inc_init[nj][np.where(edge_lab == 4)[0]] = 1 # met
    inc_init[nj][np.where(edge_lab == 2)[0]] = 1 # met

    return inc_init, edge_lab

if __name__ == '__main__':
    main()


