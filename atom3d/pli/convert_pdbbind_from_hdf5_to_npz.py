import os, sys
import pickle
import pandas as pd
import numpy as np
import argparse

sys.path.append('../..')
import atom3d.util.splits as splits
import atom3d.util.datatypes as dt

from rdkit import Chem


### --- HELPER FUNCTIONS ---

def read_labels(labels_filename, pdbcodes):
    """
    Return a pandas DataFrame containing labels for all pdbs with header
    <pdb> and <label>.
    """
    print('Reading labels from file', labels_filename)
    labels_df = pd.read_csv(labels_filename, delimiter=',', engine='python').dropna()
    return labels_df[labels_df.pdb.isin(pdbcodes)].reset_index(drop=True)


def read_split(split_filename):
    """
    Return a list of pdb codes included in the split.
    """
    print('Reading split from file', split_filename)
    with open(split_filename, 'r') as f:
        pdbcodes = [t.strip() for t in f.readlines()]
    return pdbcodes

def read_structures(struct_filename, pdbcodes):
    """
    Return a pandas DataFrame containing all structures.
    """
    print('Reading structures from file', struct_filename)
    data_df = pd.read_hdf(struct_filename, 'structures')
    data_df = data_df[data_df.ensemble.isin(pdbcodes)]
    return data_df

def load_data(labels_filename, struct_filename, split_filename):
    
    # Read PDB codes for the respective split
    all_pdb_codes = read_split(split_filename)
    # Read the labels
    labels_df = read_labels(labels_filename, all_pdb_codes)
    # Some PDB codes might not have labels, so we need to prune them
    pdb_codes = labels_df.pdb.unique()   
    # Read the structures for the pruned PDB codes
    struct_df = read_structures(struct_filename, pdb_codes)
    
    return pdb_codes, struct_df, labels_df


# --- THE DATASET CLASS ---

class MoleculesDataset():
    """Internal data set, including coordinates."""

    def __init__(self, labels_filename, struct_filename, split_filename, name='molecules'):
        """Initializes a data set.
        
        Args:
            labels_filename (str): CSV file with label data.
            struct_filename (str): HDF5 file with coordinates.
            split_filename (str): Text file with PDB codes.
            name (str, opt.): Name of the dataset. Default: 'molecules'.
        
        """
       
        pdb_codes, struct_df, labels_df = load_data(labels_filename, struct_filename, split_filename)

        pte = Chem.GetPeriodicTable()

        self.num_atoms = []
        self.charges   = []
        self.positions = []
        self.index     = []
        self.data      = []
        self.data_keys = [k for k in labels_df.keys()[1:]] # 0th key is pdb code 

        for code in struct_df.ensemble.unique():
    
            new_struct = struct_df[struct_df.ensemble==code]
            new_labels = labels_df[labels_df.pdb==code]
            new_values = [ new_labels[col].item() for col in self.data_keys ]
            new_atnums = [ pte.GetAtomicNumber(e.title()) for e in new_struct.element ]
            conf_coord = dt.get_coordinates_from_df(new_struct)

            self.num_atoms.append(len(new_struct))
            self.index.append(code)
            self.charges.append(new_atnums)
            self.positions.append(conf_coord)
            self.data.append(new_values) 
    
        return
    
    
    def __len__(self):
        """Provides the number of molecules in a data set"""
        
        return len(self.index)

    
    def __getitem__(self, idx):
        """Provides a molecule from the data set.
        
        Args:
            idx (int): The index of the desired element.

        Returns:
            sample (dict): The name of a property as a key and the property itself as a value.
        
        """
        
        sample = {'index': self.index[idx],\
                  'num_atoms': self.num_atoms[idx],\
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
        # Add the data from the CSV file (dynamically)
        for ip,prop in enumerate(self.data_keys):
            selected_data = [self.data[idx] for idx in indices]
            locals()[prop] = [col[ip] for col in selected_data]
            # Use only those quantities that are of one of the defined data types
            if datatypes is not None and np.array(locals()[prop]).dtype in datatypes:
                save_dict[prop] = locals()[prop]
        
        # Add the data from the SDF file
        save_dict['num_atoms'] = num_atoms
        save_dict['charges']   = charges
        save_dict['positions'] = positions

        # Save as a compressed array 
        np.savez_compressed(filename,**save_dict)
        
        return    


# --- CONVERSION ---

def convert_hdf5_to_npz(in_dir_name, out_dir_name, split_dir_name, datatypes=None):
    """Converts a data set given as hdf5 to npz train/validation/test sets.
        
    Args:
        in_dir_name (str): NAme of the input directory.
        out_dir_name (Str): Name of the output directory.
        split_indices (list): List of int lists [test_indices, vali_indices, train_indices]

    Returns:
        ds (MoleculesDataset): The internal data set with all processed information.
        
    """
    
    csv_file = in_dir_name+'/pdbbind_refined_set_labels.csv'
    hdf_file = in_dir_name+'/pdbbind_3dcnn.h5'

    split_tr = split_dir_name+'/core_split/train_random.txt'
    split_va = split_dir_name+'/core_split/val_random.txt'
    split_te = split_dir_name+'/core_split/test.txt'

    # Create the internal data sets
    ds_tr = MoleculesDataset(csv_file, hdf_file, split_tr)
    ds_va = MoleculesDataset(csv_file, hdf_file, split_va)
    ds_te = MoleculesDataset(csv_file, hdf_file, split_te)

    print('Training: %i molecules. Validation: %i molecules. Test: %i molecules.'%(len(ds_tr),len(ds_va),len(ds_te)))
    
    # Make a directory
    try:
        os.mkdir(out_dir_name)
    except FileExistsError:
        pass

    # Save the data sets as compressed numpy files
    te_file_name = out_dir_name+'/test.npz'
    va_file_name = out_dir_name+'/valid.npz'
    tr_file_name = out_dir_name+'/train.npz'
    ds_tr.write_compressed(te_file_name, datatypes=datatypes )
    ds_va.write_compressed(va_file_name, datatypes=datatypes )
    ds_te.write_compressed(tr_file_name, datatypes=datatypes )
        
    return ds_tr, ds_va, ds_te


############
# - MAIN - #

if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument('in_dir', type=str, help='directory with the raw data')
    parser.add_argument('out_dir', type=str, help='directory to write npz files')
    parser.add_argument('-i', dest='idx_dir', type=str, default=None, help='directory from which to read split indices') 
    args = parser.parse_args()
    
    cormorant_datatypes = ['float64', 'float32', 'float16', 'int64', 'int32', 'int16', 'int8', 'uint8', 'bool']

    ds_tr, ds_va, ds_te = convert_hdf5_to_npz(args.in_dir, args.out_dir, args.idx_dir, datatypes=cormorant_datatypes)


