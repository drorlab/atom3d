import numpy as numpy
import pandas as pd
import sys
sys.path.append('..')
from util import datatypes as dt
from util import file as fi
import torch
import torch.nn.functional as F
from scipy.spatial import KDTree

# these include co-crystallized metals with >5 occurrences in PDBBind
atoms = ['C', 'H', 'O', 'N', 'S', 'ZN', 'NA', 'FE', 'CA', 'MN', 'NI', 'CO', 'MG', 'CU', 'X']
atom_int_dict = dict(zip(atoms, range(len(atoms))))

def prot_df_to_graph(df, edge_dist_cutoff=5):

	num_nodes = df.shape[0]
	pos = torch.Tensor(df[['x', 'y', 'z']].to_numpy())

	kd_tree = KDTree(pos)
	edge_tuples = list(kd_tree.query_pairs(edge_dist_cutoff))
	edges = torch.LongTensor(edge_tuples).t().contiguous()

	elems = torch.LongTensor([atom_int_dict.get(e, atom_int_dict['X']) for e in df['element']])
	feats = F.one_hot(elems, num_classes=len(atom_int_dict))

	return feats, edges, pos