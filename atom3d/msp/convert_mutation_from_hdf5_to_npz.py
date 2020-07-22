import argparse
import os

import numpy as np
import pandas as pd
import scipy as sp
import scipy.spatial
from rdkit import Chem

import atom3d.util.datatypes as dt
import atom3d.util.shard as shard

pte = Chem.GetPeriodicTable()



# --- HELPER FUNCTIONS ---


def select_environment(df,chain,resid,dist):
    """
    Selects a region of protein coordinates within a certain distance from the alpha carbon of the mutated residue.

    Args:
        df: data frame
        chain: name of the chain with the mutated residue
        resid: residue number of the mutated residue
        dist: distance from the alpha carbon of the mutated residue

    Returns:
        key pts (int[]): indices of selected protein coordinates
    """

    mutated = df[df.chain == chain][df.residue == resid]
    mut_c_a = mutated[mutated.name == 'CA']
    protein = df

    # extract coordinates
    muta_coords = np.array([mut_c_a.x, mut_c_a.y, mut_c_a.z]).T
    prot_coords = np.array([protein.x, protein.y, protein.z]).T

    # Select the environment around mutated residues
    kd_tree = sp.spatial.KDTree(prot_coords)
    key_pts = kd_tree.query_ball_point(muta_coords, r=dist, p=2.0)
    key_pts = np.unique([k for l in key_pts for k in l])

    new_df = pd.concat([ protein.iloc[key_pts] ], ignore_index=True)

    return new_df



# --- THE DATASET CLASS ---


class MoleculesDataset():
    """Internal data set, including coordinates."""

    def __init__(self, struct_filename, name='molecules', cutoff=None, num_sampled_shards=None, max_num_atoms=None):
        """Initializes a data set.
        
        Args:
            struct_filename (str): HDF5 file with coordinates.
            labels_filename (str): CSV file with label data.
            split_filename (str): Text file with PDB codes.
            name (str, opt.): Name of the dataset. Default: 'molecules'.
        
        """
        
        print('Loading',name,'set')
        self.name = name
        
        sharded_ds = shard.load_sharded(struct_filename)
        num_shards = sharded_ds.get_num_shards()
        
        self.num_atoms = []
        self.symbols   = []
        self.charges   = []
        self.positions = []
        self.index     = []
        self.data      = []
        self.data_keys = ['label'] # only one property here
        
        # Define indices and subsample if necessary 
        shard_indices = np.arange(num_shards)
        if num_sampled_shards is not None and num_sampled_shards < num_shards:
            shard_indices = np.random.choice(shard_indices, size=num_sampled_shards, replace=False, p=None)
        
        for shard_idx in shard_indices:
                
            struct_df = sharded_ds.read_shard(shard_idx)
            labels_df = pd.read_hdf(sharded_ds._get_shard(shard_idx), 'labels')
            ensembles = labels_df['ensemble']

            for i, code in enumerate(ensembles):

                new_struct = struct_df[struct_df.ensemble==code]
                new_labels = labels_df['label'][i]

                muta_chain = labels_df['chain'][i]
                muta_resid = labels_df['residue'][i]

                # select the local environment of the mutated residue
                sel_struct = select_environment(new_struct,muta_chain,muta_resid,cutoff)
                print(code, len(new_struct),len(sel_struct))

                # move on with the next structure if this one is too large
                if max_num_atoms is not None and len(sel_struct) > max_num_atoms:
                    continue

                subunits = sel_struct.subunit.unique()
                for j, sub in enumerate(subunits):

                    sub_df = sel_struct[sel_struct.subunit == sub]
                    sub_df = sub_df.reset_index(drop=True)

                    # get element symbols
                    new_symbols = [ elem.title() for elem in sub_df.element ]

                    # get atomic numbers
                    new_atnums  = np.array([ pte.GetAtomicNumber(e.title()) for e in sub_df.element ])
                    # extract coordinates
                    conf_coord = dt.get_coordinates_from_df(sub_df)
        
                    # append everything
                    self.symbols.append(new_symbols)
                    self.charges.append(new_atnums)
                    self.positions.append(conf_coord)
                    self.num_atoms.append(len(new_atnums))

                self.data.append(new_labels) # There will be twice as many structures as labels (order matters!!!)
    
        return
    
    
    def __len__(self):
        """Provides the number of molecules in a data set"""
        
        return len(self.num_atoms)

    
    def __getitem__(self, idx):
        """Provides a molecule from the data set.
        
        Args:
            idx (int): The index of the desired element.

        Returns:
            sample (dict): The name of a property as a key and the property itself as a value.
        
        """
        
        sample = {'num_atoms': self.num_atoms[idx],\
                  'symbols': self.symbols[idx],\
                  'charges': self.charges[idx],\
                  'positions': self.positions[idx],\
                  'data': self.data[int(np.floor(idx/2))]}

        return sample
    
    
    def write_compressed(self, filename):
        """Writes (a subset of) the data set as compressed numpy arrays.

        Args:
            filename (str):  The name of the output file. 

        """

        # Define which molecules to use 
        # (counting indices of processed data set)
        indices = np.arange(len(self))
        # All charges and position arrays have the same size
        # (the one of the biggest molecule)
        size = np.max( self.num_atoms )
        # Initialize arrays
        num_atoms = np.zeros(len(indices))
        labels    = np.zeros(len(indices))
        charges   = np.zeros([len(indices),size])
        positions = np.zeros([len(indices),size,3])
        # For each molecule ...
        for j,idx in enumerate(indices):
            # load the data
            sample = self[idx]
            # assign per-molecule data
            labels[j]    = sample['data']
            num_atoms[j] = sample['num_atoms']
            # ... and for each atom:
            for ia in range(sample['num_atoms']):
                charges[j,ia] = sample['charges'][ia]
                positions[j,ia,0] = sample['positions'][ia][0] 
                positions[j,ia,1] = sample['positions'][ia][1] 
                positions[j,ia,2] = sample['positions'][ia][2]

        # Merge pairs
        print(labels.shape,charges.shape,positions.shape)
        labels = labels[0::2]
        charges = np.array([np.concatenate((charges[i],charges[i+1])) for i in indices[0::2]])
        positions = np.array([np.concatenate((positions[i],positions[i+1])) for i in indices[0::2]])
        print(labels.shape,charges.shape,positions.shape)
        
        # Create a dictionary with all the values to save
        save_dict = {}
        save_dict['label']     = labels
        save_dict['charges']   = charges
        save_dict['positions'] = positions

        # Save as a compressed array 
        np.savez_compressed(filename,**save_dict)
        
        return
        



# --- CONVERSION ---


def convert_hdf5_to_npz(in_dir_name, out_dir_name, cutoff=None,
                        num_sampled_shards=None, max_num_atoms=None):
    """Converts a data set given as hdf5 to npz train/validation/test sets.
        
    Args:
        in_dir_name (str): Name of the input directory.
        out_dir_name (Str): Name of the output directory.

    Returns:
        ds (MoleculesDataset): The internal data set with all processed information.
        
    """

    tr_env_fn = in_dir_name+'/split/pairs_train@40'
    va_env_fn = in_dir_name+'/split/pairs_val@40'
    te_env_fn = in_dir_name+'/split/pairs_test@40'

    # Create the internal data sets
    ds_tr = MoleculesDataset(tr_env_fn, cutoff=cutoff, name='training')
    ds_va = MoleculesDataset(va_env_fn, cutoff=cutoff, name='validation')
    ds_te = MoleculesDataset(te_env_fn, cutoff=cutoff, name='test')

    print('Training: %i molecules. Validation: %i molecules. Test: %i molecules.'%(len(ds_tr),len(ds_va),len(ds_te)))

    # Make a directory
    try:
        os.mkdir(out_dir_name)
    except FileExistsError:
        pass

    # Save the data sets as compressed numpy files
    tr_file_name = out_dir_name+'/train.npz'
    va_file_name = out_dir_name+'/valid.npz'
    te_file_name = out_dir_name+'/test.npz'
    ds_tr.write_compressed(tr_file_name)
    ds_va.write_compressed(va_file_name)
    ds_te.write_compressed(te_file_name)

    return ds_tr, ds_va, ds_te




############
# - MAIN - #

if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument('in_dir', type=str, help='directory with the raw data')
    parser.add_argument('out_dir', type=str, help='directory to write npz files')
    parser.add_argument('--cutoff', dest='cutoff', type=int, default=None, help='cutoff to select region around mutated residue')
    args = parser.parse_args()
    
    ds_tr, ds_va, ds_te = convert_hdf5_to_npz(args.in_dir, args.out_dir, cutoff=args.cutoff)


