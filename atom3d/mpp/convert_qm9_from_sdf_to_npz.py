import os, sys
import click
import pickle
import pandas as pd
import numpy as np
import argparse
sys.path.append('../..')
import atom3d.util.splits as split
import atom3d.util.datatypes as dt

from rdkit import Chem



class MoleculesDataset():
    """Internal data set, including coordinates."""

    def __init__(self, csv_file, sdf_file, name='molecules'):
        """Initializes a data set from a CSV file.
        
        Args:
            csv_file (str): CSV file with label data.
            sdf_file (str): SDF file with coordinates.
            name (str, opt.): Name of the dataset. Default: 'molecules'.
        
        """
        
        df = pd.read_csv(csv_file)
        self.raw_data = [ df[col] for col in df.keys() ]
        self.raw_mols = dt.read_sdf_to_mol(sdf_file, sanitize=False)
        
        # Simple sanity check:
        # Is the number of molecules the same in both files?
        assert len(self.raw_mols) == len(self.raw_data[0])

        self.num_atoms = []
        self.charges   = []
        self.positions = []
        self.index     = []
        self.data      = []
        self.data_keys = [k for k in df.keys()]

        for im, m in enumerate(self.raw_mols):
    
            if m is None: 
                print('Molecule',im+1,'could not be processed.')
                continue

            new_atnums = np.array([a.GetAtomicNum() for a in m.GetAtoms()])
            conf_coord = dt.get_coordinates_of_conformer(m)

            self.num_atoms.append(m.GetNumAtoms())
            self.index.append(im+1)
            self.charges.append(new_atnums)
            self.positions.append(conf_coord)
            self.data.append([ col[im] for col in self.raw_data])
    
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
        index     = np.zeros(len(indices)) # this is the index from the original data set
        num_atoms = np.zeros(len(indices))
        charges   = np.zeros([len(indices),size])
        positions = np.zeros([len(indices),size,3])
        # For each molecule ...
        for j,idx in enumerate(indices):
            # load the data
            sample = self[idx]
            # assign per-molecule data
            index[j]     = sample['index']
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
            locals()[prop] = [col[ip] for col in self.data]
            # Use only those quantities that are of one of the defined data types
            if datatypes is not None and np.array(locals()[prop]).dtype in datatypes:
                save_dict[prop] = locals()[prop]
        # Add the data from the SDF file
        save_dict['index']     = index
        save_dict['num_atoms'] = num_atoms
        save_dict['charges']   = charges
        save_dict['positions'] = positions

        # Save as a compressed array 
        np.savez_compressed(filename,**save_dict)
        
        return


def convert_sdfcsv_to_npz(csv_file, sdf_file, out_dir_name, split_indices=None, datatypes=None):
    """Converts a data set given as CSV list and SDF coordinates to npz train/validation/test sets.
        
    Args:
        csv_file (str): CSV file with labels.
        sdf_file (str): SDF file with coordinates.
        out_dir_name (Str): Name of the output directory.
        split_indices (list): List of int lists [test_indices, vali_indices, train_indices]

    Returns:
        ds (MoleculesDataset): The internal data set with all processed information.
        
    """
    
    seed = 42

    # Create the internal data set
    ds = MoleculesDataset(csv_file,sdf_file)

    # Define indices to split the data set
    if split_indices is None:
        test_indices, vali_indices, train_indices = split.random_split(len(ds),vali_split=0.1,test_split=0.1,random_seed=seed)
    else:
        test_indices, vali_indices, train_indices = split_indices
    print('Training: %i molecules. Validation: %i molecules. Test: %i molecules.'%(len(train_indices),len(vali_indices),len(test_indices)))
    
    # Make a directory
    try:
        os.mkdir(out_dir_name)
    except FileExistsError:
        pass

    # Save the indices for the split
    np.savetxt(out_dir_name+'/indices_test.dat',test_indices)
    np.savetxt(out_dir_name+'/indices_valid.dat',vali_indices)
    np.savetxt(out_dir_name+'/indices_train.dat',train_indices)

    # Save the data sets as compressed numpy files
    test_file_name  = out_dir_name+'/test.npz'
    vali_file_name  = out_dir_name+'/valid.npz'
    train_file_name = out_dir_name+'/train.npz'
    if len(test_indices) > 0: ds.write_compressed(test_file_name, indices=test_indices, datatypes=datatypes )
    if len(vali_indices) > 0: ds.write_compressed(vali_file_name, indices=vali_indices, datatypes=datatypes )
    if len(train_indices) > 0: ds.write_compressed(train_file_name, indices=train_indices, datatypes=datatypes )
        
    return ds


############
# - MAIN - #

if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument('csv_file', type=str, help='label file in CSV format')
    parser.add_argument('sdf_file', type=str, help='structure file in SDF format')
    parser.add_argument('out_dir', type=str, help='directory to write npz files')
    args = parser.parse_args()
    
    cormorant_datatypes = ['float64', 'float32', 'float16', 'int64', 'int32', 'int16', 'int8', 'uint8', 'bool']

    ds = convert_sdfcsv_to_npz(args.csv_file, args.sdf_file, args.out_dir, datatypes=cormorant_datatypes)


