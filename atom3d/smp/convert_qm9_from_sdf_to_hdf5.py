import os, sys
import pickle
import pandas as pd
import numpy as np
import argparse
sys.path.append('../..')
import atom3d.util.splits as splits
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

        self.index     = []
        self.data      = []
        self.data_keys = [k for k in df.keys()]
        self.mol_dfs   = []

        for im, m in enumerate(self.raw_mols):
    
            if m is None: 
                print('Molecule',im+1,'could not be processed.')
                continue
            
            self.index.append(im+1)
            self.data.append([ col[im] for col in self.raw_data])
            new_mol_df = dt.mol_to_df(m,addHs=False,structure=df['mol_id'][im])
            self.mol_dfs.append(new_mol_df)

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
                  'mol_df': self.mol_dfs[idx],\
                  'data': self.data[idx]}

        return sample
    
    
    def write_hdf5(self, filename, indices=None, datatypes=None):
        """Writes (a subset of) the data set as compressed numpy arrays.

        Args:
            filename (str):  The name of the output file. 
            indices (int[]): The indices of the molecules to write data for.

        """

        # Define which molecules to use 
        # (counting indices of processed data set)
        if indices is None:
            indices = np.arange(len(self))

        # Save labels in a csv file
        out_data = pd.DataFrame([self.data[idx] for idx in indices], columns=self.data_keys)
        out_data.to_csv(filename+'.csv', index=False)

        # Save structures as a python data frame in an hdf5 file 
        combined = pd.concat([self.mol_dfs[idx] for idx in indices])
        combined.to_hdf(filename+'.h5', 'structures', mode='w')
 
        return


def convert_sdfcsv_to_hdf5(in_dir_name, out_dir_name, split_indices=None, datatypes=None):
    """Converts a data set given as CSV list and SDF coordinates to HDF5 train/validation/test sets.
        
    Args:
        in_dir_name (str): Name of the input directory.
        out_dir_name (Str): Name of the output directory.
        split_indices (list): List of int lists [test_indices, vali_indices, train_indices]

    Returns:
        ds (MoleculesDataset): The internal data set with all processed information.
        
    """
    
    seed = 42

    csv_file = in_dir_name+'/gdb9_with_cv_atom.csv'
    sdf_file = in_dir_name+'/gdb9.sdf'
    unc_file = in_dir_name+'/uncharacterized.txt'

    # Create the internal data set
    ds = MoleculesDataset(csv_file,sdf_file)

    # Load the list of molecules to ignore 
    with open(unc_file, 'r') as f:
        exclude = [int(x.split()[0]) for x in f.read().split('\n')[9:-2]]
    assert len(exclude) == 3054 

    # Define indices to split the data set
    if split_indices is None:
        test_indices, vali_indices, train_indices = splits.random_split(len(ds),vali_split=0.1,test_split=0.1,random_seed=seed,exclude=exclude)
    else:
        test_indices, vali_indices, train_indices = split_indices
    print('Training: %i molecules. Validation: %i molecules. Test: %i molecules.'%(len(train_indices),len(vali_indices),len(test_indices)))
    
    # Make a directory
    try:
        os.mkdir(out_dir_name)
    except FileExistsError:
        pass

    # Save the indices for the split
#    np.savetxt(out_dir_name+'/indices_test.dat',test_indices,fmt='%1d')
#    np.savetxt(out_dir_name+'/indices_valid.dat',vali_indices,fmt='%1d')
#    np.savetxt(out_dir_name+'/indices_train.dat',train_indices,fmt='%1d')

    # Save the data sets as compressed numpy files
    test_file_name  = out_dir_name+'/test'
    vali_file_name  = out_dir_name+'/valid'
    train_file_name = out_dir_name+'/train'
    if len(test_indices) > 0: ds.write_hdf5(test_file_name, indices=test_indices, datatypes=datatypes )
    if len(vali_indices) > 0: ds.write_hdf5(vali_file_name, indices=vali_indices, datatypes=datatypes )
    if len(train_indices) > 0: ds.write_hdf5(train_file_name, indices=train_indices, datatypes=datatypes )
        
    return ds


############
# - MAIN - #

if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument('in_dir', type=str, help='directory with the raw data')
    parser.add_argument('out_dir', type=str, help='directory to write npz files')
    parser.add_argument('-i', dest='idx_dir', type=str, default=None, help='directory from which to read split indices') 
    args = parser.parse_args()
    
    cormorant_datatypes = ['float64', 'float32', 'float16', 'int64', 'int32', 'int16', 'int8', 'uint8', 'bool']

    if args.idx_dir is not None:
        test_indices  = np.loadtxt(args.idx_dir+'/indices_test.dat',dtype=int)
        vali_indices  = np.loadtxt(args.idx_dir+'/indices_valid.dat',dtype=int)
        train_indices = np.loadtxt(args.idx_dir+'/indices_train.dat',dtype=int)
        split = [test_indices, vali_indices, train_indices]
    else:
        split = None

    ds = convert_sdfcsv_to_hdf5(args.in_dir, args.out_dir, split_indices=split, datatypes=cormorant_datatypes)


