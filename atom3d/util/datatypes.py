"""Methods to convert between different file formats."""
import collections as col
import gzip
import os
import re

import Bio.PDB
from rdkit import Chem

import pandas as pd
import numpy as np


patterns = {
    'pdb': 'pdb[0-9]*$',
    'pdb.gz': 'pdb[0-9]*\.gz$',
    'mmcif': '(mm)?cif$',
    'sharded': '@[0-9]+',
}

_regexes = {k: re.compile(v) for k, v in patterns.items()}


def is_sharded(f):
    """If file is in sharded format."""
    return _regexes['sharded'].search(f)


def is_pdb(f):
    """If file is in pdb format."""
    return _regexes['pdb'].search(f)


def is_mmcif(f):
    """If file is in mmcif format."""
    return _regexes['mmcif'].search(f)


def is_pdb_gz(f):
    """If file is in mmcif format."""
    return _regexes['pdb.gz'].search(f)


def read_any(f, name=None):
    """Read file into biopython structure."""
    if is_pdb(f):
        return read_pdb(f, name)
    elif is_pdb_gz(f):
        return read_pdb_gz(f, name)
    elif is_mmcif(f):
        return read_mmcif(f, name)
    else:
        raise ValueError(f"Unrecognized filetype for {f:}")


def read_pdb_gz(pdb_gz_file, name=None):
    if name is None:
        name = os.path.basename(pdb_gz_file)
    parser = Bio.PDB.PDBParser(QUIET=True)
    bp = parser.get_structure(
        name, gzip.open(pdb_gz_file, mode='rt', encoding='latin1'))
    return bp


def read_pdb(pdb_file, name=None):
    """Load pdb file in to biopython representation."""
    if name is None:
        name = os.path.basename(pdb_file)
    parser = Bio.PDB.PDBParser(QUIET=True)
    bp = parser.get_structure(name, pdb_file)
    return bp


def read_mmcif(mmcif_file, name=None):
    """Load mmcif file in to biopython representation."""
    if name is None:
        os.path.basename(mmcif_file)
    parser = Bio.PDB.MMCIFParser(QUIET=True)
    return parser.get_structure(name, mmcif_file)


def write_pdb(out_file, structure, **kwargs):
    """Write a biopython structure to a pdb file."""
    io = Bio.PDB.PDBIO()
    io.set_structure(structure)
    io.save(out_file, **kwargs)
    return


def write_mmcif(out_file, structure):
    """Write a biopython structure to an mmcif file."""
    io = Bio.PDB.MMCIFIO()
    io.set_structure(structure)
    io.save(out_file)
    return


def read_xyz(filename):
    """Read GDB9-style xyz file."""
    with open(filename) as xyzfile:
        # Extract number of atoms
        num_at = int(xyzfile.readline())
        #print('Reading file', filename, 'with', num_at, 'atoms.')
        # Read header
        header = xyzfile.readline()
        # Initialize lists, arrays
        elements    = []
        charges     = np.zeros(num_at)
        coordinates = np.zeros([num_at,3])
        # Iterate through all atoms and read info
        for i in range(num_at):
            line = xyzfile.readline()
            el,x,y,z,q = line.split()
            elements.append(el)
            charges[i] = q
            coordinates[i] = np.array([x,y,z],dtype=float)
        # Read footer
        footer = xyzfile.readline()
        # Read SMILES and InChi
        smiles1, smiles2 = xyzfile.readline().split()
        inchi1,  inchi2  = xyzfile.readline().split()
    # Construct the dictionary
    data = {'smiles':smiles1, 'inchi':inchi1,
            'header':header, 'footer':footer,
            'elements':elements, 'charges':charges,
            'coordinates':coordinates}
    return data


def bp_to_df(bp):
    """Convert biopython representation to pandas dataframe representation."""
    df = col.defaultdict(list)
    for atom in Bio.PDB.Selection.unfold_entities(bp, 'A'):
        residue = atom.get_parent()
        chain = residue.get_parent()
        model = chain.get_parent()
        df['structure'].append(bp._id)
        df['model'].append(model.serial_num)
        df['chain'].append(chain.id)
        df['hetero'].append(residue.id[0])
        df['insertion_code'].append(residue.id[2])
        df['residue'].append(residue.id[1])
        df['segid'].append(residue.segid)
        df['resname'].append(residue.resname)
        df['altloc'].append(atom.altloc)
        df['occupancy'].append(atom.occupancy)
        df['bfactor'].append(atom.bfactor)
        df['x'].append(atom.coord[0])
        df['y'].append(atom.coord[1])
        df['z'].append(atom.coord[2])
        df['element'].append(atom.element)
        df['name'].append(atom.name)
        df['fullname'].append(atom.fullname)
        df['serial_number'].append(atom.serial_number)
    df = pd.DataFrame(df)
    return df


def df_to_bp(df_in):
    """Convert dataframe representaion to biopython representation."""
    all_structures = df_to_bps(df_in)
    if len(all_structures) > 1:
        raise RuntimeError('More than one structure in provided dataframe.')
    return all_structures[0]


def df_to_bps(df_in):
    """Convert dataframe representation to biopython representations."""
    df = df_in.copy()
    all_structures = []
    for (structure, s_atoms) in split_df(df_in):
        new_structure = Bio.PDB.Structure.Structure(structure)
        for (model, m_atoms) in df.groupby(['model']):
            new_model = Bio.PDB.Model.Model(model)
            for (chain, c_atoms) in m_atoms.groupby(['chain']):
                new_chain = Bio.PDB.Chain.Chain(chain)
                for (residue, r_atoms) in c_atoms.groupby(
                        ['residue', 'insertion_code']):
                    # Take first atom as representative for residue values.
                    rep = r_atoms.iloc[0]
                    new_residue = Bio.PDB.Residue.Residue(
                        (rep['hetero'], rep['residue'], rep['insertion_code']),
                        rep['resname'], rep['segid'])
                    for row, atom in r_atoms.iterrows():
                        new_atom = Bio.PDB.Atom.Atom(
                            atom['name'],
                            [atom['x'], atom['y'], atom['z']],
                            atom['bfactor'],
                            atom['occupancy'],
                            atom['altloc'],
                            atom['fullname'],
                            atom['serial_number'],
                            atom['element'])
                        new_residue.add(new_atom)

                    new_chain.add(new_residue)
                new_model.add(new_chain)
            new_structure.add(new_model)
        all_structures.append(new_structure)
    return all_structures


def split_df(df):
    return [(x, y) for x, y in df.groupby('structure')]


def merge_dfs(dfs):
    return pd.concat(dfs).reset_index(drop=True)



def bp_from_xyz_dict(data,struct_name='structure'):
    """Construct a biopython structure from xyz data (stored in a dict)."""
    # Read info from dictionary
    elements = data['elements']
    charges = data['charges']
    coordinates = data['coordinates']
    # Create a residue
    # (each small molecule is counted as just one residue)
    r = Bio.PDB.Residue.Residue((' ',1,' '),'res',0)
    # Iterate through all atoms and collect info
    for i in range(len(charges)):
        atom_name = elements[i]+str(i)
        position  = coordinates[i]
        full_name = elements[i]+str(i)
        b_factor  = 0.0
        occupancy = 1.0
        alt_loc   = ' '
        serial_n  = i
        element   = elements[i]
        # Create an atom with the provided information
        a = Bio.PDB.Atom.Atom(atom_name,
                              position,
                              b_factor,
                              occupancy,
                              alt_loc,
                              full_name,
                              serial_n,
                              element=element)
        # Add the atom to the residue
        r.add(a)
    # Create one chain and add the residue
    c = Bio.PDB.Chain.Chain('A')
    c.add(r)
    # Create one model and add the chain
    m = Bio.PDB.Model.Model(0)
    m.add(c)
    # Create one structure and add the model
    s = Bio.PDB.Structure.Structure(struct_name)
    s.add(m)
    return s


def read_sdf_to_mol(sdf_file,sanitize=True):

    suppl = Chem.SDMolSupplier(sdf_file,sanitize=sanitize)
    molecules = [mol for mol in suppl]

    return molecules


def get_coordinates_of_conformer(mol):
    """Reads the coordinates of the conformer

    Args:
        mol (Mol): Molecule in RDKit format.

    Returns:
        xyz (float array): Coordinates

    """

    symb = [a.GetSymbol() for a in mol.GetAtoms()]
    conf = mol.GetConformer()
    xyz  = np.empty([mol.GetNumAtoms(),3])

    for ia, name in enumerate(symb):
        position = conf.GetAtomPosition(ia)
        xyz[ia]  = np.array([position.x, position.y, position.z])

    return xyz


def get_connectivity_matrix(mol):
    """Generates the connection matrix from a molecule.

    Args:
        mol (Mol): a molecule in RDKit format

    Returns:
        connect_matrix (2D numpy array): connectivity matrix

    """

    # Initialization
    num_at = mol.GetNumAtoms()
    connect_matrix = np.zeros([num_at,num_at],dtype=int)

    # Go through all atom pairs and check for bonds between them
    for a in mol.GetAtoms():
        for b in mol.GetAtoms():
            bond = mol.GetBondBetweenAtoms(a.GetIdx(),b.GetIdx())
            if bond is not None:
                connect_matrix[a.GetIdx(),b.GetIdx()] = 1

    return connect_matrix


def get_bonds_matrix(mol):
    """Provides bond types encoded as single (1.0). double (2.0), triiple (3.0), and aromatic (1.5).

    Args:
        mol (Mol): a molecule in RDKit format

    Returns:
        connect_matrix (2D numpy array): connectivity matrix

    """

    # Initialization
    num_at = mol.GetNumAtoms()
    bonds_matrix = np.zeros([num_at,num_at])

    # Go through all atom pairs and check for bonds between them
    for a in mol.GetAtoms():
        for b in mol.GetAtoms():
            bond = mol.GetBondBetweenAtoms(a.GetIdx(),b.GetIdx())
            if bond is not None:
                bt = bond.GetBondTypeAsDouble()
                bonds_matrix[a.GetIdx(),b.GetIdx()] = bt

    return bonds_matrix

