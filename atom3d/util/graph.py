import numpy as np
import scipy.spatial as ss
import torch

import atom3d.util.datatypes as dt

# PDB atom names -- these include co-crystallized metals
prot_atoms = ['C', 'H', 'O', 'N', 'S', 'P', 'ZN', 'NA', 'FE', 'CA', 'MN', 'NI', 'CO', 'MG', 'CU', 'CL', 'SE', 'F', 'X']
# RDKit molecule atom names
mol_atoms = ['C', 'N', 'O', 'S', 'F', 'Si', 'P', 'Cl', 'Br', 'Mg', 'Na',
             'Ca', 'Fe', 'As', 'Al', 'I', 'B', 'V', 'K', 'Tl', 'Yb',
             'Sb', 'Sn', 'Ag', 'Pd', 'Co', 'Se', 'Ti', 'Zn', 'H',  # H?
             'Li', 'Ge', 'Cu', 'Au', 'Ni', 'Cd', 'In', 'Mn', 'Zr',
             'Cr', 'Pt', 'Hg', 'Pb', 'Unknown']


def prot_df_to_graph(df, edge_dist_cutoff=4.5):
    """
    Converts protein in dataframe representation to a graph compatible with Pytorch-Geometric

    Args:
        df (DataFrame): protein in dataframe format
        edge_dist_cutoff (float): max distance to define an edge between two atoms

    Returns:
        node_feats (LongTensor): features for each node, one-hot encoded by element
        edge_feats (LongTensor): features for each node, one-hot encoded by element
        edges (LongTensor): edges in COO format
        node_pos (FloatTensor): x-y-z coordinates of each node
    """

    node_pos = torch.FloatTensor(df[['x', 'y', 'z']].to_numpy())

    kd_tree = ss.KDTree(node_pos)
    edge_tuples = list(kd_tree.query_pairs(edge_dist_cutoff))
    edges = torch.LongTensor(edge_tuples).t().contiguous()

    node_feats = torch.FloatTensor([one_of_k_encoding_unk(e, prot_atoms) for e in df['element']])
    edge_feats = torch.FloatTensor(
        [1.0 / (np.linalg.norm(node_pos[i] - node_pos[j]) + 1e-5) for i, j in edge_tuples]).view(-1, 1)
    # feats = F.one_hot(elems, num_classes=len(atom_int_dict))

    return node_feats, edges, edge_feats, node_pos


def prot_df_to_res_graph(df, edge_dist_cutoff=9.0):
    """
    Converts protein in dataframe representation to a graph compatible with Pytorch-Geometric

    Args:
        df (DataFrame): protein in dataframe format
        edge_dist_cutoff (float): max distance to define an edge between two residues (by CA atom)

    Returns:
        node_feats (LongTensor): features for each node, one-hot encoded by element
        edge_feats (LongTensor): features for each node, one-hot encoded by element
        edges (LongTensor): edges in COO format
        node_pos (FloatTensor): x-y-z coordinates of each node
    """
    # PDB atom names -- these include co-crystallized metals with >5 occurrences in PDBBind
    residues = ['ALA', 'CYS', 'ASP', 'GLU', 'PHE', 'GLY', 'HIS', 'ILE', 'LYS', 'LEU', 'MET', 'ASN', 'PRO', 'GLN', 'ARG',
                'SER', 'THR', 'VAL', 'TRP', 'TYR']

    df = df[df['name'] == 'CA']
    node_pos = torch.FloatTensor(df[['x', 'y', 'z']].to_numpy())

    kd_tree = ss.KDTree(node_pos)
    edge_tuples = list(kd_tree.query_pairs(edge_dist_cutoff))
    edges = torch.LongTensor(edge_tuples).t().contiguous()

    node_feats = torch.FloatTensor([one_of_k_encoding(e, residues) for e in df['resname']])
    edge_feats = torch.FloatTensor([1.0 / np.linalg.norm(node_pos[i] - node_pos[j]) for i, j in edge_tuples]).view(-1,
                                                                                                                   1)
    # feats = F.one_hot(elems, num_classes=len(atom_int_dict))

    return node_feats, edges, edge_feats, node_pos


def mol_to_graph(mol):
    """
    Converts Mol object to a graph compatible with Pytorch-Geometric

    Args:
        mol (Mol): RDKit Mol object

    Returns:
        node_feats (LongTensor): features for each node, one-hot encoded by element
        edge_feats (LongTensor): features for each node, one-hot encoded by element
        edges (LongTensor): edges in COO format
        node_pos (FloatTensor): x-y-z coordinates of each node
    """
    node_pos = torch.FloatTensor(dt.get_coordinates_of_conformer(mol))
    bonds = dt.get_bonds_matrix(mol)
    edge_tuples = np.argwhere(bonds)
    edges = torch.LongTensor(edge_tuples).t().contiguous()

    node_feats = torch.FloatTensor([one_of_k_encoding_unk(a.GetSymbol(), mol_atoms) for a in mol.GetAtoms()])
    # edge_feats = torch.FloatTensor([one_of_k_encoding(bonds[i,j], [1.0, 2.0, 3.0, 1.5]) for i,j in edge_tuples])
    edge_feats = torch.FloatTensor([bonds[i, j] for i, j in edge_tuples]).view(-1, 1)

    return node_feats, edges, edge_feats, node_pos


def combine_graphs(graph1, graph2, edges_between=True):
    node_feats1, edges1, edge_feats1, pos1 = graph1
    node_feats2, edges2, edge_feats2, pos2 = graph2

    dummy_node_feats1 = torch.zeros(pos1.shape[0], node_feats2.shape[1])
    dummy_node_feats2 = torch.zeros(pos2.shape[0], node_feats1.shape[1])
    node_feats1 = torch.cat((node_feats1, dummy_node_feats1), dim=1)
    node_feats2 = torch.cat((dummy_node_feats2, node_feats2), dim=1)

    edges2 += pos1.shape[0]

    node_pos = torch.cat((pos1, pos2), dim=0)
    node_feats = torch.cat((node_feats1, node_feats2), dim=0)

    if edges_between:
        edges_between, edge_feats_between = edges_between_graphs(pos1, pos2)
        edge_feats = torch.cat((edge_feats1, edge_feats2, edge_feats_between), dim=0)
        edges = torch.cat((edges1, edges2, edges_between), dim=1)
    else:
        edge_feats = torch.cat((edge_feats1, edge_feats2), dim=0)
        edges = torch.cat((edges1, edges2), dim=1)

    return node_feats, edges, edge_feats, node_pos


def edges_between_graphs(pos1, pos2):
    tree1 = ss.KDTree(pos1)
    tree2 = ss.KDTree(pos2)
    res = tree1.query_ball_tree(tree2, r=4.5)
    edges = []
    edge_weights = []
    for i, contacts in enumerate(res):
        if len(contacts) == 0:
            continue
        for j in contacts:
            edges.append((i, j + pos1.shape[0]))
            edge_weights.append(np.linalg.norm(pos1[i] - pos2[j]))

    edges = torch.LongTensor(edges).t().contiguous()
    edge_weights = torch.FloatTensor(edge_weights).view(-1, 1)
    return edges, edge_weights


# adapted from DeepChem repository:
def one_of_k_encoding(x, allowable_set):
    if x not in allowable_set:
        raise Exception("input {0} not in allowable set{1}:".format(x, allowable_set))
    return list(map(lambda s: x == s, allowable_set))


def one_of_k_encoding_unk(x, allowable_set):
    """Maps inputs not in the allowable set to the last element."""
    if x not in allowable_set:
        x = allowable_set[-1]
    return list(map(lambda s: x == s, allowable_set))
