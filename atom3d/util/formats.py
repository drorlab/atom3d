"""Methods to convert between different file formats."""
import collections as col
import gzip
import os
import re

import Bio.PDB.Atom
import Bio.PDB.Chain
import Bio.PDB.Model
import Bio.PDB.Residue
import Bio.PDB.Structure
import numpy as np
import pandas as pd

patterns = {
    'pdb': 'pdb[0-9]*$',
    'pdb.gz': 'pdb[0-9]*\.gz$',
    'mmcif': '(mm)?cif$',
    'sharded': '@[0-9]+',
    'sdf': 'sdf[0-9]*$',
    'xyz': 'xyz[0-9]*$',
}

_regexes = {k: re.compile(v) for k, v in patterns.items()}


def is_sharded(f):
    """If file is in sharded format."""
    return _regexes['sharded'].search(str(f))


def is_pdb(f):
    """If file is in pdb format."""
    return _regexes['pdb'].search(str(f))


def is_mmcif(f):
    """If file is in mmcif format."""
    return _regexes['mmcif'].search(str(f))


def is_sdf(f):
    """If file is in sdf format."""
    return _regexes['sdf'].search(str(f))


def is_pdb_gz(f):
    """If file is in mmcif format."""
    return _regexes['pdb.gz'].search(str(f))

def is_xyz(f):
    """If file is in xyz format."""
    return _regexes['xyz'].search(str(f))


def read_any(f, name=None):
    """Read file into biopython structure."""
    if is_pdb(f):
        return read_pdb(f, name)
    elif is_pdb_gz(f):
        return read_pdb_gz(f, name)
    elif is_mmcif(f):
        return read_mmcif(f, name)
    elif is_sdf(f):
        return read_sdf(f, name)
    elif is_xyz(f):
        return read_xyz(f, name)
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


def read_sdf(sdf_file, sanitize=True, add_hs=False, remove_hs=True):
    dflist = []
    molecules = read_sdf_to_mol(sdf_file, sanitize=sanitize,
                                add_hs=add_hs, remove_hs=remove_hs)
    for im,m in enumerate(molecules):
        if m is not None:
            df = mol_to_df(m, residue=im, 
                           ensemble = m.GetProp("_Name"), 
                           structure = m.GetProp("_Name"), 
                           model = m.GetProp("_Name"))
            dflist.append(df)
    bp = df_to_bp(merge_dfs(dflist))
    return bp


def read_sdf_multi(sdf_files, sanitize=True, add_hs=False, remove_hs=True):
    dflist = []
    for sdf_file in sdf_files:
        molecules = read_sdf_to_mol(sdf_file, sanitize=sanitize,
                                    add_hs=add_hs, remove_hs=remove_hs)
        for im,m in enumerate(molecules):
            if m is not None:
                df = mol_to_df(m, residue=im,
                               ensemble = m.GetProp("_Name"),
                               structure = m.GetProp("_Name"),
                               model = m.GetProp("_Name"))
                dflist.append(df)
    bp = df_to_bp(merge_dfs(dflist))
    return bp


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


def read_xyz_to_df(inputfile, gdb_data=False):
    """Read an XYZ file (optionally with GDB9-specific data)"""
    with open(inputfile) as f:
        # Reading number of atoms in the molecule
        num_atoms = int(f.readline().strip())
        # Loading GDB ID and label data
        line_labels = f.readline().strip().split('\t')
        name = line_labels[0]
        if gdb_data: data = [float(ll) for ll in line_labels[1:]]
        # Skip atom data (will be read using pandas below)
        for n in range(num_atoms): f.readline()
        # Harmonic vibrational frequencies
        if gdb_data: 
            freq = [float(ll) for ll in f.readline().strip().split('\t')]
        # SMILES and InChI
        if gdb_data: smiles = f.readline().strip().split('\t')[0]
        if gdb_data: inchi  = f.readline().strip().split('\t')[0]
    # Define columns: element, x, y, z, Mulliken charges (GDB only)
    columns = ['element','x', 'y', 'z']
    if gdb_data: columns += ['charge']
    # Load atom information
    molecule = pd.read_table(inputfile, names=columns,
                             skiprows=2, delim_whitespace=True)
    molecule = molecule[:num_atoms]
    # Name the dataframe
    molecule.name = name
    molecule.index.name = name
    # return molecule info
    if gdb_data: 
        return molecule, data, freq, smiles, inchi
    else:
        return molecule


def read_xyz(xyz_file, name=None, gdb=False):
    """Load xyz file in to biopython representation."""
    # Load the xyz file into a dataframe
    if gdb:
        df, data, freq, smiles, inchi = read_xyz_to_df(xyz_file, gdb_data=True)
    else:
        df = read_xyz_to_df(xyz_file)
    if name is not None: df.index.name = name
    # Make up atom names
    elements = df['element'].unique()
    el_count = {}
    for e in elements: 
        el_count[e] = 0
    new_name = []
    for el in df['element']:
        el_count[e] += 1
        new_name.append('%s%i'%(el,el_count[e]))
    # Fill additional fields
    df['ensemble'] = [df.name.replace(' ','_')]*len(df)
    df['subunit'] = [0]*len(df)
    df['structure'] = [df.name.replace(' ','_')]*len(df)
    df['model'] = [0]*len(df)
    df['chain'] = ['L']*len(df)
    df['hetero'] = ['']*len(df)
    df['insertion_code'] = ['']*len(df)
    df['residue'] = [1]*len(df)
    df['segid'] = ['LIG']*len(df)
    df['resname'] = ['LIG']*len(df)
    df['altloc'] = ['']*len(df)
    df['occupancy'] = [1.]*len(df)
    df['bfactor'] = [0.]*len(df)
    df['name'] = new_name
    df['fullname'] = new_name
    df['serial_number'] = range(len(df))
    # Convert to biopython representation
    bp = df_to_bp(df)
    if gdb:
        return bp, data, freq, smiles, inchi
    else:
        return bp


def bp_to_df(bp):
    """Convert biopython representation to pandas dataframe representation."""
    df = col.defaultdict(list)
    for atom in Bio.PDB.Selection.unfold_entities(bp, 'A'):
        residue = atom.get_parent()
        chain = residue.get_parent()
        model = chain.get_parent()
        df['ensemble'].append(bp.get_id())
        df['subunit'].append(0)
        df['structure'].append(bp.get_id())
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
    for (structure, s_atoms) in split_df(df_in, ['ensemble', 'structure']):
        new_structure = Bio.PDB.Structure.Structure(structure[1])
        for (model, m_atoms) in df.groupby(['model']):
            new_model = Bio.PDB.Model.Model(model)
            for (chain, c_atoms) in m_atoms.groupby(['chain']):
                new_chain = Bio.PDB.Chain.Chain(chain)
                for (residue, r_atoms) in c_atoms.groupby(
                        ['hetero', 'residue', 'insertion_code']):
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


def split_df(df, key):
    return [(x, y) for x, y in df.groupby(key)]


def merge_dfs(dfs):
    return pd.concat(dfs).reset_index(drop=True)


def bp_from_xyz_dict(data, struct_name='structure'):
    """Construct a biopython structure from xyz data (stored in a dict)."""
    # Read info from dictionary
    elements = data['elements']
    charges = data['charges']
    coordinates = data['coordinates']
    # Create a residue
    # (each small molecule is counted as just one residue)
    r = Bio.PDB.Residue.Residue((' ', 1, ' '), 'res', 0)
    # Iterate through all atoms and collect info
    for i in range(len(charges)):
        atom_name = elements[i] + str(i)
        position = coordinates[i]
        full_name = elements[i] + str(i)
        b_factor = 0.0
        occupancy = 1.0
        alt_loc = ' '
        serial_n = i
        element = elements[i]
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


def read_sdf_to_mol(sdf_file, sanitize=True, add_hs=False, remove_hs=True):
    from rdkit import Chem
    suppl = Chem.SDMolSupplier(sdf_file, sanitize=sanitize, removeHs=remove_hs)
    molecules = [mol for mol in suppl]
    if add_hs:
        molecules = [Chem.AddHs(mol, addCoords=True) for mol in suppl]

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
    xyz = np.empty([mol.GetNumAtoms(), 3])

    for ia, name in enumerate(symb):
        position = conf.GetAtomPosition(ia)
        xyz[ia] = np.array([position.x, position.y, position.z])

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
    connect_matrix = np.zeros([num_at, num_at], dtype=int)

    # Go through all atom pairs and check for bonds between them
    for a in mol.GetAtoms():
        for b in mol.GetAtoms():
            bond = mol.GetBondBetweenAtoms(a.GetIdx(), b.GetIdx())
            if bond is not None:
                connect_matrix[a.GetIdx(), b.GetIdx()] = 1

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
    bonds_matrix = np.zeros([num_at, num_at])

    # Go through all atom pairs and check for bonds between them
    for a in mol.GetAtoms():
        for b in mol.GetAtoms():
            bond = mol.GetBondBetweenAtoms(a.GetIdx(), b.GetIdx())
            if bond is not None:
                bt = bond.GetBondTypeAsDouble()
                bonds_matrix[a.GetIdx(), b.GetIdx()] = bt

    return bonds_matrix


def mol_to_df(mol, add_hs=False, structure=None, model=None, ensemble=None, residue=999):
    """
    Convert Mol object to dataframe format (with PDB columns)
    """
    from rdkit import Chem
    df = col.defaultdict(list)
    if add_hs:
        mol = Chem.AddHs(mol, addCoords=True)
    conf = mol.GetConformer()
    for i, a in enumerate(mol.GetAtoms()):
        position = conf.GetAtomPosition(i)
        df['ensemble'].append(ensemble)
        df['structure'].append(structure)
        df['model'].append(model)
        df['chain'].append('LIG')
        df['hetero'].append('')
        df['insertion_code'].append('')
        df['residue'].append(residue)
        df['segid'].append('')
        df['resname'].append('LIG')
        df['altloc'].append('')
        df['occupancy'].append(1)
        df['bfactor'].append(0)
        df['x'].append(position.x)
        df['y'].append(position.y)
        df['z'].append(position.z)
        df['element'].append(a.GetSymbol())
        df['name'].append("%s%i"%(a.GetSymbol(),i+1))
        df['fullname'].append("%s%i"%(a.GetSymbol(),i+1))
        df['serial_number'].append(i)
    df = pd.DataFrame(df)
    return df


def get_coordinates_from_df(df):
    xyz = np.empty([len(df), 3])

    xyz[:, 0] = np.array(df.x)
    xyz[:, 1] = np.array(df.y)
    xyz[:, 2] = np.array(df.z)

    return xyz

