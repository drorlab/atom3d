import argparse
import os

import numpy as np
import pandas as pd
from rdkit import Chem

import atom3d.shard.shard as shard
import atom3d.util.formats as dt

label_res_dict={0:'HIS',1:'LYS',2:'ARG',3:'ASP',4:'GLU',5:'SER',6:'THR',7:'ASN',8:'GLN',9:'ALA',10:'VAL',11:'LEU',12:'ILE',13:'MET',14:'PHE',15:'TYR',16:'TRP',17:'PRO',18:'GLY',19:'CYS'}
res_label_dict={'HIS':0,'LYS':1,'ARG':2,'ASP':3,'GLU':4,'SER':5,'THR':6,'ASN':7,'GLN':8,'ALA':9,'VAL':10,'LEU':11,'ILE':12,'MET':13,'PHE':14,'TYR':15,'TRP':16,'PRO':17,'GLY':18,'CYS':19}
bb_atoms = ['N', 'CA', 'C']

pte = Chem.GetPeriodicTable()



# --- THE DATASET CLASS ---


class MoleculesDataset():
    """Internal data set, including coordinates."""

    def __init__(self, struct_filename, name='molecules', shuffle=False, num_sampled_shards=None, max_num_atoms=None):
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
        self.data      = []
        self.data_keys = ['residue'] # only one property here
        
        # Define indices and subsample if necessary 
        shard_indices = np.arange(num_shards)
        if num_sampled_shards is not None and num_sampled_shards < num_shards:
            shard_indices = np.random.choice(shard_indices, size=num_sampled_shards, replace=False, p=None)
        
        for i, shard_idx in enumerate(shard_indices):

            print('Processing shard',shard_idx,' -- ',i,'/',len(shard_indices))
            
            s = sharded_ds.read_shard(shard_idx)
            
            if shuffle:
                groups = [df for _, df in s.groupby(['ensemble', 'subunit'])]
                random.shuffle(groups)
                s = pd.concat(groups).reset_index(drop=True)
                
            for ens, new_struct in s.groupby(['ensemble', 'subunit']):
                # get the label (residue to predict)
                subunit = ens[1]
                res_name = subunit.split('_')[-1]
                label = res_label_dict[res_name]
                # get element symbols
                new_symbols = [ elem.title() for elem in new_struct.element ]
                # move on with the next structure if this one is too large
                if max_num_atoms is not None and len(new_symbols) > max_num_atoms:
                    continue
                # get atomic numbers
                new_atnums  = np.array([ pte.GetAtomicNumber(e.title()) for e in new_struct.element ])
                # extract coordinates
                conf_coord = dt.get_coordinates_from_df(new_struct)
                # append everything
                self.data.append([label])
                self.symbols.append(new_symbols)
                self.charges.append(new_atnums)
                self.positions.append(conf_coord)
                self.num_atoms.append(len(new_atnums))
    
        return
    
    
    def __len__(self):
        """Provides the number of molecules in a data set"""
        
        return len(self.data)

    
    def __getitem__(self, idx):
        """Provides a molecule from the data set.
        
        Args:
            idx (int): The index of the desired element.

        Returns:
            sample (dict): The name of a property as a key and the property itself as a value.
        
        """
        
        sample = {'num_atoms': self.num_atoms[idx],\
                  'charges': self.charges[idx],\
                  'positions': self.positions[idx],\
                  'data': self.data[idx]}

        return sample
    
    
    def write_compressed(self, filename, indices=None, datatypes=None):
        """Writes (a subset of) the data set as compressed numpy arrays.

        Args:
            filename (str):  The name of the output file. 
            indices (int[]): The indices of the molecules to write data for.

        """
        
        print('Writing',self.name,'set')

        # Define which molecules to use 
        # (counting indices of processed data set)
        if indices is None:
            indices = np.arange(len(self))
        # All charges and position arrays have the same size
        # (the one of the biggest molecule)
        size = np.max( self.num_atoms )
        # Initialize arrays
        num_atoms = np.zeros(len(indices))
        charges   = np.zeros([len(indices),size])
        positions = np.zeros([len(indices),size,3])
        # For each molecule ...
        for j,idx in enumerate(indices):
            # load the data
            sample = self[idx]
            # assign per-molecule data
            num_atoms[j] = sample['num_atoms']
            # ... and for each atom:
            for ia in range(sample['num_atoms']):
                charges[j,ia] = sample['charges'][ia]
                positions[j,ia,0] = sample['positions'][ia][0] 
                positions[j,ia,1] = sample['positions'][ia][1] 
                positions[j,ia,2] = sample['positions'][ia][2]

        # Create a dictionary with all the values to save
        save_dict = {}
        # Add the label data (dynamically)
        for ip,prop in enumerate(self.data_keys):
            selected_data = [self.data[idx] for idx in indices]
            locals()[prop] = [col[ip] for col in selected_data]
            # Use only those quantities that are of one of the defined data types
            if datatypes is not None and np.array(locals()[prop]).dtype in datatypes:
                save_dict[prop] = locals()[prop]

        # Add the structural data
        save_dict['num_atoms'] = num_atoms
        save_dict['charges']   = charges
        save_dict['positions'] = positions

        # Save as a compressed array 
        np.savez_compressed(filename,**save_dict)
        
        return



# --- CONVERSION ---


def convert_hdf5_to_npz(in_dir_name, out_dir_name, datatypes=None, shuffle=False, 
                        num_sampled_shards_tr=10, num_sampled_shards_va=4, num_sampled_shards_te=5, 
                        max_num_atoms=None):
    """Converts a data set given as hdf5 to npz train/validation/test sets.
        
    Args:
        in_dir_name (str): NAme of the input directory.
        out_dir_name (Str): Name of the output directory.

    Returns:
        ds (MoleculesDataset): The internal data set with all processed information.
        
    """

    tr_env_fn = in_dir_name+'/split/train_envs@1000'
    va_env_fn = in_dir_name+'/split/val_envs@100'
    te_env_fn = in_dir_name+'/split/test_envs@100'

    # Create the internal data sets
    ds_tr = MoleculesDataset(tr_env_fn, shuffle=shuffle, max_num_atoms=max_num_atoms, num_sampled_shards=int(num_sampled_shards_tr), name='training')
    ds_va = MoleculesDataset(va_env_fn, shuffle=False,   max_num_atoms=max_num_atoms, num_sampled_shards=int(num_sampled_shards_va), name='validation')
    ds_te = MoleculesDataset(te_env_fn, shuffle=False,   max_num_atoms=max_num_atoms, num_sampled_shards=int(num_sampled_shards_te), name='test')

    types_tr = np.unique(np.concatenate(ds_tr.charges))
    types_te = np.unique(np.concatenate(ds_te.charges))
    types_va = np.unique(np.concatenate(ds_va.charges))

    assert np.all(types_tr==types_te) and np.all(types_va==types_te)

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
    ds_tr.write_compressed(tr_file_name, datatypes=datatypes )
    ds_va.write_compressed(va_file_name, datatypes=datatypes )
    ds_te.write_compressed(te_file_name, datatypes=datatypes )

    return ds_tr, ds_va, ds_te



############
# - MAIN - #

if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument('in_dir', type=str, help='directory with the raw data')
    parser.add_argument('out_dir', type=str, help='directory to write npz files')
    parser.add_argument('--maxnumat', dest='maxnumat', type=float, default=None, help='drop all structures with more than this number of atoms')
    parser.add_argument('--numshards_tr', dest='numshards_tr', type=int, default=10, help='number of shards to sample for training')
    parser.add_argument('--numshards_va', dest='numshards_va', type=int, default=4,  help='number of shards to sample for validation')
    parser.add_argument('--numshards_te', dest='numshards_te', type=int, default=5,  help='number of shards to sample for testing')
    args = parser.parse_args()
    
    cormorant_datatypes = ['float64', 'float32', 'float16', 'int64', 'int32', 'int16', 'int8', 'uint8', 'bool']

    ds_tr, ds_va, ds_te = convert_hdf5_to_npz(args.in_dir, args.out_dir, datatypes=cormorant_datatypes, max_num_atoms=args.maxnumat, 
                                              num_sampled_shards_tr=args.numshards_tr, num_sampled_shards_va=args.numshards_va, num_sampled_shards_te=args.numshards_te)


