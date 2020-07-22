import argparse
import os

import numpy as np
import pandas as pd
import scipy as sp
import scipy.spatial
from rdkit import Chem

import atom3d.util.datatypes as dt


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


def select_binding_pocket(df,dist=6):
    """
    Selects a region of protein coordinates within a certain distance from the ligand. 
    
    Args:
        prot_coords: protein coordinates
        lig_coords:  ligand coordinates
        dist: distance from the ligand [in Anstroms]
        
    Returns:
        key pts (int[]): indices of selected protein coordinates
    """

    ligand  = df[df.chain=='LIG']
    protein = df[df.chain!='LIG']
    lig_coords  = np.array([ligand.x, ligand.y, ligand.z]).T
    prot_coords = np.array([protein.x, protein.y, protein.z]).T
    
    # Select the binding pocket
    kd_tree = sp.spatial.KDTree(prot_coords)
    key_pts = kd_tree.query_ball_point(lig_coords, r=dist, p=2.0)
    #key_pts = set([k for l in key_pts for k in l])
    key_pts = np.unique([k for l in key_pts for k in l])
    
    new_df = pd.concat([protein.iloc[key_pts], ligand], ignore_index=True)
    
    return new_df


def valid_elements(symbols,reference):
    """Tests a list for elements that are not in the reference.

    Args:
        symbols (list): The list whose elements to check.
        reference (list): The list containing all allowed elements.

    Returns:
        valid (bool): True if symbols only contains elements from the reference.

    """

    valid = True

    if reference is not None:
        for sym in symbols:
            if sym not in reference:
                valid = False

    return valid



# --- THE DATASET CLASS ---


class MoleculesDataset():
    """Internal data set, including coordinates."""

    def __init__(self, labels_filename, struct_filename, split_filename, name='molecules', 
                 drop_hydrogen=False, cutoff=None, max_num_atoms=None, elements=None, element_dict=None):
        """Initializes a data set.
        
        Args:
            struct_filename (str): HDF5 file with coordinates.
            labels_filename (str): CSV file with label data.
            split_filename (str): Text file with PDB codes.
            name (str, opt.): Name of the dataset. Default: 'molecules'.
        
        """
        
        # Read PDB codes for the respective split
        all_pdb_codes = read_split(split_filename)
        # Read the labels
        labels_df = read_labels(labels_filename, all_pdb_codes)
        # Some PDB codes might not have labels, so we need to prune them
        pdb_codes = labels_df.pdb.unique()   
        # Read the structures for the pruned PDB codes
        struct_df = read_structures(struct_filename, pdb_codes)
    
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
            
            # select the binding pocket
            if cutoff is None:
                sel_struct = new_struct
            else:
                sel_struct = select_binding_pocket(new_struct,dist=cutoff)

            # get element symbols
            if element_dict is None:
                sel_symbols = [ e.title() for e in sel_struct.element ]
            else:
                sel_symbols = [ element_dict[e.title()] for e in sel_struct.element ]
            # move on with the next structure if this one contains unwanted elements
            if not valid_elements(sel_symbols,elements):
                continue

            # get atomic numbers
            sel_atnums  = np.array([ pte.GetAtomicNumber(e.title()) for e in sel_struct.element ])
            # extract coordinates
            conf_coord = dt.get_coordinates_from_df(sel_struct)
            # select heavy (=non-H) atoms
            if drop_hydrogen:
                heavy_atom = np.array(sel_atnums)!=1
                sel_atnums = sel_atnums[heavy_atom]
                conf_coord = conf_coord[heavy_atom]
            # move on with the next structure if this one is too large
            if max_num_atoms is not None and len(sel_atnums) > max_num_atoms:
                continue

            self.index.append(code)
            self.data.append(new_values)
            self.charges.append(sel_atnums)
            self.positions.append(conf_coord)
            self.num_atoms.append(len(sel_atnums))

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

def convert_hdf5_to_npz(in_dir_name, out_dir_name, split_dir_name, datatypes=None, droph=False, 
                        cutoff=None, max_num_atoms=None, elements=None, element_dict=None):
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

    split_tr = split_dir_name+'/train.txt'
    split_va = split_dir_name+'/val.txt'
    split_te = split_dir_name+'/test.txt'

    # Create the internal data sets
    ds_tr = MoleculesDataset(csv_file, hdf_file, split_tr, drop_hydrogen=droph, cutoff=cutoff, 
                             max_num_atoms=max_num_atoms, elements=elements, element_dict=element_dict)
    ds_va = MoleculesDataset(csv_file, hdf_file, split_va, drop_hydrogen=droph, cutoff=cutoff, 
                             max_num_atoms=max_num_atoms, elements=elements, element_dict=element_dict)
    ds_te = MoleculesDataset(csv_file, hdf_file, split_te, drop_hydrogen=droph, cutoff=cutoff, 
                             max_num_atoms=max_num_atoms, elements=elements, element_dict=element_dict)

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
    parser.add_argument('-i', dest='idx_dir', type=str, default=None, help='directory from which to read split indices') 
    parser.add_argument('--drop_h', dest='drop_h', action='store_true', help='drop hydrogen atoms')
    parser.add_argument('--cutoff', dest='cutoff', type=float, default=None, help='cut off the protein beyond this distance around the ligand [Angstrom]')
    parser.add_argument('--maxnumat', dest='maxnumat', type=float, default=None, help='drop all structures with more than this number of atoms')
    args = parser.parse_args()
    
    elements_pdbbind = ['H','C','N','O','S','Zn','Cl','F','P','Mg'] #,'Br','Ca','Mn','I']
    cormorant_datatypes = ['float64', 'float32', 'float16', 'int64', 'int32', 'int16', 'int8', 'uint8', 'bool']
    element_dict = None

    ds_tr, ds_va, ds_te = convert_hdf5_to_npz(args.in_dir, args.out_dir, args.idx_dir, 
                                              datatypes=cormorant_datatypes, droph=args.drop_h, cutoff=args.cutoff, 
                                              max_num_atoms=args.maxnumat, elements=elements_pdbbind, element_dict=element_dict)


