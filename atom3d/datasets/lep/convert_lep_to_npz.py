import argparse
import os

import numpy as np
import pandas as pd
import scipy as sp
import scipy.spatial
from rdkit import Chem

import atom3d.util.formats as dt
import atom3d.shard.shard as sh
import atom3d.shard.shard_ops as sho



### --- HELPER FUNCTIONS ---


def load_data(sharded_name):
    """Iterate through shards to obtain structures and labels."""

    input_sharded = sh.Sharded.load(sharded_name)

    struct = []
    labels = []

    for shard_num in range(input_sharded.get_num_shards()):
        struct.append(input_sharded.read_shard(shard_num))
        labels.append(input_sharded.read_shard(shard_num, 'labels'))

    struct_df = pd.concat(struct)
    labels_df = pd.concat(labels)

    return struct_df, labels_df


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

    ligand  = df[df.chain=='L']
    protein = df[df.chain!='L']
    lig_coords  = np.array([ligand.x, ligand.y, ligand.z]).T
    prot_coords = np.array([protein.x, protein.y, protein.z]).T
    
    # Select the binding pocket
    kd_tree = sp.spatial.KDTree(prot_coords)
    key_pts = kd_tree.query_ball_point(lig_coords, r=dist, p=2.0)
    #key_pts = set([k for l in key_pts for k in l])
    key_pts = np.unique([k for l in key_pts for k in l])
    
    new_df = pd.concat([protein.iloc[key_pts], ligand], ignore_index=False).sort_index()

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

    def __init__(self, sharded_name, name='molecules',
                 drop_hydrogen=False, cutoff=None, max_num_atoms=None, elements=None, element_dict=None):
        """Initializes a data set.
        
        Args:
            sharded_name (str): Identifier for the sharded dataset..
            name (str, opt.): Name of the dataset. Default: 'molecules'.
        
        """

        # Read structures and labels
        struct_df, labels_df = load_data(sharded_name)

        pte = Chem.GetPeriodicTable()

        self.num_atoms = []
        self.charges   = []
        self.positions = []
        self.index     = []
        self.data      = []
        self.data_keys = ['label'] # 0th key is ensemble code 
        self.active_su = [] # is the atom part of the active subunit

        for code in struct_df.ensemble.unique():

            new_struct = struct_df[struct_df.ensemble==code]
            new_labels = labels_df[labels_df.ensemble==code]
            new_labels = new_labels.reset_index()
            new_values = [ int(new_labels.at[0,'label']=='A') ]
            
            # select the binding pocket
            if cutoff is None:
                sel_struct = new_struct
            else:
                sel_struct = select_binding_pocket(new_struct,dist=cutoff)
            # get atoms belonging to the active structure
            sel_active = np.array([su[-7:] == '_active' for su in sel_struct.subunit], dtype=int)
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
                sel_active = sel_active[heavy_atom]
                conf_coord = conf_coord[heavy_atom]
            # move on with the next structure if this one is too large
            if max_num_atoms is not None and len(sel_atnums) > max_num_atoms:
                continue

            self.index.append(code)
            self.data.append(new_values)
            self.charges.append(sel_atnums)
            self.positions.append(conf_coord)
            self.num_atoms.append(len(sel_atnums))
            self.active_su.append(sel_active)

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
                  'data': self.data[idx],\
                  'active': self.active_su[idx]}

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
        active_su = np.zeros([len(indices),size])
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
                active_su[j,ia] = sample['active'][ia]
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
        save_dict['active']    = active_su
        
        # Save as a compressed array 
        np.savez_compressed(filename,**save_dict)

        return



# --- CONVERSION ---

def convert_hdf5_to_npz(in_dir_name, out_dir_name, datatypes=None, droph=False, 
                        cutoff=None, max_num_atoms=None, elements=None, element_dict=None):
    """Converts a data set given as hdf5 to npz train/validation/test sets.
        
    Args:
        in_dir_name (str): Name of the input directory.
        out_dir_name (Str): Name of the output directory.
        split_indices (list): List of int lists [test_indices, vali_indices, train_indices]

    Returns:
        ds (MoleculesDataset): The internal data set with all processed information.
        
    """
    
    # Create the internal data sets
    ds_tr = MoleculesDataset(in_dir_name+'/pairs_train@10', drop_hydrogen=droph, cutoff=cutoff, 
                             max_num_atoms=max_num_atoms, elements=elements, element_dict=element_dict)
    print('Training: %i molecules.'%(len(ds_tr)), flush=True)
    ds_va = MoleculesDataset(in_dir_name+'/pairs_val@10',   drop_hydrogen=droph, cutoff=cutoff, 
                             max_num_atoms=max_num_atoms, elements=elements, element_dict=element_dict)
    print('Validation: %i molecules.'%(len(ds_va)), flush=True)
    ds_te = MoleculesDataset(in_dir_name+'/pairs_test@10',  drop_hydrogen=droph, cutoff=cutoff, 
                             max_num_atoms=max_num_atoms, elements=elements, element_dict=element_dict)
    print('Test: %i molecules.'%(len(ds_te)), flush=True)
    
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
    parser.add_argument('--drop_h', dest='drop_h', action='store_true', help='drop hydrogen atoms')
    parser.add_argument('--cutoff', dest='cutoff', type=float, default=None, help='cut off the protein beyond this distance around the ligand [Angstrom]')
    parser.add_argument('--maxnumat', dest='maxnumat', type=float, default=None, help='drop all structures with more than this number of atoms')
    args = parser.parse_args()
    
    elements = ['H','C','N','O','S','Cl','F']
    datatypes = ['float64', 'float32', 'float16', 'int64', 'int32', 'int16', 'int8', 'uint8', 'bool']
    element_dict = None

    ds_tr, ds_va, ds_te = convert_hdf5_to_npz(args.in_dir, args.out_dir,  
                                              datatypes=datatypes, droph=args.drop_h, cutoff=args.cutoff, 
                                              max_num_atoms=args.maxnumat, elements=elements, element_dict=element_dict)


