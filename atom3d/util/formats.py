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


# -- MANIPULATING DATAFRAMES --


def split_df(df, key):
    """
    Split dataframe containing structure(s) based on specified key. Most commonly used to split by ensemble (`key="ensemble"`) or subunit (`key=["ensemble", "subunit"]`).

    :param df: Molecular structure(s) in ATOM3D dataframe format.
    :type df: pandas.DataFrame
    :param key: key on which to split dataframe. To split on multiple keys, provide all keys in a list. Must be compatible with dataframe hierarchy, i.e. ensemble > subunit > structure > model > chain.
    :type key: Union[str, list[str]]

    :return: List of tuples containing keys and corresponding sub-dataframes.
    :rtypes: list[tuple]
    """
    return [(x, y) for x, y in df.groupby(key)]


def merge_dfs(dfs):
    """Combine a list of dataframes into a single dataframe. Assumes dataframes contain the same columns."""
    return pd.concat(dfs).reset_index(drop=True)


# -- CONVERTING INTERNAL FORMATS --


def bp_to_df(bp):
    """Convert biopython representation to ATOM3D dataframe representation.

    :param bp: Molecular structure in Biopython representation.
    :type bp: Bio.PDB.Structure

    :return: Molecular structure in ATOM3D dataframe format.
    :rtype: pandas.DataFrame
    """
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
    """Convert ATOM3D dataframe representation to biopython representation. Assumes dataframe contains only one structure.

    :param df_in: Molecular structure in ATOM3D dataframe format.
    :type df_in: pandas.DataFrame

    :return: Molecular structure in BioPython format.
    :rtype: Bio.PDB.Structure
    """
    all_structures = df_to_bps(df_in)
    if len(all_structures) > 1:
        raise RuntimeError('More than one structure in provided dataframe.')
    return all_structures[0]


def df_to_bps(df_in):
    """Convert ATOM3D dataframe representation containing multiple structures to list of Biopython structures. Assumes different structures are specified by `ensemble` and `structure` columns of dataframe.

    :param df_in: Molecular structures in ATOM3D dataframe format.
    :type df_in: pandas.DataFrame

    :return : List of molecular structures in BioPython format.
    :rtype: list[Bio.PDB.Structure]
    """
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


# -- READING FILES -- #


## general reader function to get a Biopython structure
#  (not suported: sharded, silent, xyz-gdb)


def read_any(f, name=None):
    """Read any ATOM3D file type into Biopython structure (compatible with pdb, pdb.gz, mmcif, sdf, xyz).

    :param f: file path
    :type f: Union[str, Path]
    :param name: optional name or identifier for structure. If None (default), use file basename.
    :type name: str

    :return: Biopython object containing structure
    :rtype: Bio.PDB.Structure
    """
    if is_pdb(f):
        return read_pdb(f, name)
    elif is_pdb_gz(f):
        return read_pdb_gz(f, name)
    elif is_mmcif(f):
        return read_mmcif(f, name)
    elif is_sdf(f):
        return read_sdf(f)
    elif is_xyz(f):
        return read_xyz(f, name)
    else:
        raise ValueError(f"Unrecognized filetype for {f:}")


## functions to check file format


patterns = {
    'pdb': r'pdb[0-9]*$',
    'pdb.gz': r'pdb[0-9]*\.gz$',
    'mmcif': r'(mm)?cif$',
    'sdf': r'sdf[0-9]*$',
    'xyz': r'xyz[0-9]*$',
    'xyz-gdb': r'xyz[0-9]*$',
    'silent': r'out$',
    'sharded': r'@[0-9]+',
}

_regexes = {k: re.compile(v) for k, v in patterns.items()}


def is_type(f, filetype):
    return _regexes[filetype].search(str(f))


def is_pdb(f):
    """Check if file is in pdb format."""
    return _regexes['pdb'].search(str(f))


def is_pdb_gz(f):
    """Check if file is in mmcif format."""
    return _regexes['pdb.gz'].search(str(f))


def is_mmcif(f):
    """Check if file is in mmcif format."""
    return _regexes['mmcif'].search(str(f))


def is_sdf(f):
    """Check if file is in sdf format."""
    return _regexes['sdf'].search(str(f))


def is_xyz(f):
    """Check if file is in xyz format."""
    return _regexes['xyz'].search(str(f))


def is_sharded(f):
    """Check if file is in sharded format."""
    return _regexes['sharded'].search(str(f))


## reader functions for specific file formats


def read_pdb(pdb_file, name=None):
    """Read pdb file into Biopython structure.

    :param pdb_file: file path
    :type pdb_file: Union[str, Path]
    :param name: optional name or identifier for structure. If None (default), use file basename.
    :type name: str

    :return: Biopython object containing structure
    :rtype: Bio.PDB.Structure
    """
    if name is None:
        name = os.path.basename(pdb_file)
    parser = Bio.PDB.PDBParser(QUIET=True)
    bp = parser.get_structure(name, pdb_file)
    return bp


def read_pdb_gz(pdb_gz_file, name=None):
    """Read pdb.gz file into Biopython structure.

    :param pdb_gz_file: file path
    :type pdb_gz_file: Union[str, Path]
    :param name: optional name or identifier for structure. If None (default), use file basename.
    :type name: str

    :return: Biopython object containing structure
    :rtype: Bio.PDB.Structure
    """
    if name is None:
        name = os.path.basename(pdb_gz_file)
    parser = Bio.PDB.PDBParser(QUIET=True)
    bp = parser.get_structure(
        name, gzip.open(pdb_gz_file, mode='rt', encoding='latin1'))
    return bp


def read_mmcif(mmcif_file, name=None):
    """Read mmCIF file into Biopython structure.

    :param mmcif_file: file path
    :type mmcif_file: Union[str, Path]
    :param name: optional name or identifier for structure. If None (default), use file basename.
    :type name: str

    :return: Biopython object containing structure
    :rtype: Bio.PDB.Structure
    """
    if name is None:
        os.path.basename(mmcif_file)
    parser = Bio.PDB.MMCIFParser(QUIET=True)
    return parser.get_structure(name, mmcif_file)


def read_sdf(sdf_file, name=None, sanitize=False, add_hs=False, remove_hs=False):
    """Read SDF file into Biopython structure.

    :param sdf_file: file path
    :type sdf_file: Union[str, Path]
    :param sanitize: sanitize structure with RDKit.
    :type sanitize: bool
    :param add_hs: add hydrogen atoms with RDKit.
    :type add_hs: bool
    :param remove_hs: remove hydrogen atoms with RDKit.
    :type remove_hs: bool

    :return: Biopython object containing structure
    :rtype: Bio.PDB.Structure
    """

    dflist = []
    molecules = read_sdf_to_mol(sdf_file, sanitize=sanitize,
                                add_hs=add_hs, remove_hs=remove_hs)
    for im,m in enumerate(molecules):
        if m is not None:
            df = mol_to_df(m, residue=im,
                           ensemble=m.GetProp("_Name"),
                           structure=m.GetProp("_Name"),
                           model=m.GetProp("_Name"))
            dflist.append(df)
    assert len(dflist) >= 1
    if len(dflist) > 1:
        bp = df_to_bp(merge_dfs(dflist))
    else:
        bp = df_to_bp(dflist[0])

    return bp


def read_sdf_to_mol(sdf_file, sanitize=False, add_hs=False, remove_hs=False):
    """Reads a list of molecules from an SDF file.

    :param add_hs: Specifies whether to add hydrogens. Defaults to False
    :type add_hs: bool
    :param remove_hs: Specifies whether to remove hydrogens. Defaults to False
    :type remove_hs: bool
    :param sanitize: Specifies whether to sanitize the molecule. Defaults to False
    :type sanitize: bool

    :return: list of molecules in RDKit format.
    :rtype: list[rdkit.Chem.rdchem.Mol]
    """
    from rdkit import Chem

    suppl = Chem.SDMolSupplier(sdf_file, sanitize=sanitize, removeHs=remove_hs)

    molecules = [mol for mol in suppl]

    if add_hs:
        for mol in molecules:
            if mol is not None:
                mol = Chem.AddHs(mol, addCoords=True)

    return molecules


def mol_to_df(mol, add_hs=False, structure=None, model=None, ensemble=None, residue=999):
    """
    Convert molecule in RDKit format to ATOM3D dataframe format, with PDB-style columns.

    :param mol: Molecule in RDKit format.
    :type mol: rdkit.Chem.rdchem.Mol

    :return: Dataframe in standard ATOM3D format.
    :rtype: pandas.DataFrame
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
        df['element'].append(a.GetSymbol().upper())
        df['serial_number'].append(i)
    df = pd.DataFrame(df)
    # Make up atom names
    elements = df['element'].unique()
    el_count = {}
    for e in elements:
        el_count[e] = 0
    new_name = []
    for el in df['element']:
        el_count[el] += 1
        new_name.append('%s%i'%(el,el_count[el]))
    df['name'] = new_name
    df['fullname'] = new_name
    return df


def read_xyz(xyz_file, name=None, gdb=False):
    """Read an XYZ file into Biopython representation (optionally with GDB9-specific data)

    :param inputfile: Path to input file in XYZ format.
    :type inputfile: Union[str, Path]
    :param gdb_data: Specifies whether to process and return GDB9-specific data.
    :type gdb_date: bool

    :return: If `gdb=False`, returns Biopython Structure object containing molecule structure. If `gdb=True`, returns tuple containing \n
        \t- bp (Bio.PDB.Structure): Biopython object containing molecule structure.\n
        \t- data (list[float]): Scalar molecular properties.\n
        \t- freq (list[float]): Harmonic vibrational frequencies (:math:`3n_{atoms}−5` or :math:`3n_{atoms}-6`, in :math:`cm^{−1}`).\n
        \t- smiles (str): SMILES string from GDB-17 and from B3LYP relaxation.\n
        \t- inchi (str): InChI string for Corina and B3LYP geometries.
    """
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
        el_count[el] += 1
        new_name.append('%s%i'%(el,el_count[el]))
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


def read_xyz_to_df(inputfile, gdb_data=False):
    """Read an XYZ file into Pandas DataFrame representation (optionally with GDB9-specific data)

    :param inputfile: Path to input file in XYZ format.
    :type inputfile: Union[str, Path]
    :param gdb_data: Specifies whether to process and return GDB9-specific data.
    :type gdb_date: bool

    :return: If `gdb=False`, returns DataFrame containing molecule structure. If `gdb=True`, returns tuple containing\n
        \t- molecule (pandas.DataFrame): Pandas DataFrame containing molecule structure.\n
        \t- data (list[float]): Scalar molecular properties. Returned only when `gdb=True`.\n
        \t- freq (list[float]): Harmonic vibrational frequencies (:math:`3n_{atoms}−5` or :math:`3n_{atoms}-6`, in :math:`cm^{−1}`).  Returned only when `gdb=True`.\n
        \t- smiles (str): SMILES string from GDB-17 and from B3LYP relaxation. Returned only when `gdb=True`.\n
        \t- inchi (str): InChI string for Corina and B3LYP geometries. Returned only when `gdb=True`.\n
    """
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
                             skiprows=2, nrows=num_atoms,
                             delim_whitespace=True)
    # Name the dataframe
    molecule.name = name
    molecule.index.name = name
    # return molecule info
    if gdb_data:
        return molecule, data, freq, smiles, inchi
    else:
        return molecule


# -- WRITING FILES --


def write_pdb(out_file, structure, **kwargs):
    """Write a biopython structure to a pdb file. This function accepts any viable arguments to Bio.PDB.PDBIO.save() as keyword arguments.

    :param out_file: Path to output PDB file.
    :type out_file: Union[str, Path]
    :param structure: Biopython object containing protein structure.
    :type structure: Bio.PDB.Structure
    """
    io = Bio.PDB.PDBIO()
    io.set_structure(structure)
    io.save(out_file, **kwargs)
    return


def write_mmcif(out_file, structure, **kwargs):
    """Write a biopython structure to an mmcif file. This function accepts any viable arguments to Bio.PDB.MMCIFIO.save() as keyword arguments.

    :param out_file: Path to output mmCIF file.
    :type out_file: Union[str, Path]
    :param structure: Biopython object containing protein structure.
    :type structure: Bio.PDB.structure
    """
    io = Bio.PDB.MMCIFIO()
    io.set_structure(structure)
    io.save(out_file)
    return


# -- CONVENIENCE FUNCTIONS AND CONSTANTS--
#    (for custom data conversions)

atomic_number = {'H': 1, 'He': 2, 'Li': 3, 'Be': 4, 'B': 5, 'C': 6, 'N': 7, 'O': 8, 'F': 9, 'Ne': 10,
                 'Na': 11, 'Mg': 12, 'Al': 13, 'Si': 14, 'P': 15, 'S': 16, 'Cl': 17, 'Ar': 18, 'K': 19, 'Ca': 20,
                 'Sc': 21, 'Ti': 22, 'V': 23, 'Cr': 24, 'Mn': 25, 'Fe': 26, 'Co': 27, 'Ni': 28, 'Cu': 29, 'Zn': 30,
                 'Ga': 31, 'Ge': 32, 'As': 33, 'Se': 34, 'Br': 35, 'Kr': 36, 'Rb': 37, 'Sr': 38, 'Y': 39, 'Zr': 40,
                 'Nb': 41, 'Mo': 42, 'Tc': 43, 'Ru': 44, 'Rh': 45, 'Pd': 46, 'Ag': 47, 'Cd': 48, 'In': 49, 'Sn': 50,
                 'Sb': 51, 'Te': 52, 'I': 53, 'Xe': 54, 'Cs': 55, 'Ba': 56, 'La': 57, 'Ce': 58, 'Pr': 59, 'Nd': 60,
                 'Pm': 61, 'Sm': 62, 'Eu': 63, 'Gd': 64, 'Tb': 65, 'Dy': 66, 'Ho': 67, 'Er': 68, 'Tm': 69, 'Yb': 70,
                 'Lu': 71, 'Hf': 72, 'Ta': 73, 'W': 74, 'Re': 75, 'Os': 76, 'Ir': 77, 'Pt': 78, 'Au': 79, 'Hg': 80,
                 'Tl': 81, 'Pb': 82, 'Bi': 83, 'Po': 84, 'At': 85, 'Rn': 86, 'Fr': 87, 'Ra': 88, 'Ac': 89, 'Th': 90,
                 'Pa': 91, 'U': 92, 'Np': 93, 'Pu': 94, 'Am': 95, 'Cm': 96, 'Bk': 97, 'Cf': 98, 'Es': 99, 'Fm': 100,
                 'Md': 101, 'No': 102, 'Lr': 103, 'Rf': 104, 'Db': 105, 'Sg': 106, 'Bh': 107, 'Hs': 108, 'Mt': 109, 'Ds': 110,
                 'Rg': 111, 'Cn': 112, 'Nh': 113, 'Fl': 114, 'Mc': 115, 'Lv': 116, 'Ts': 117, 'Og': 118}


def get_coordinates_from_df(df):
    """Extract XYZ coordinates from molecule in dataframe format.

    :param df: Dataframe containing molecular structure. Must have columns named `x`, `y`, and `z`.
    :type df: pandas.DataFrame

    :return: XYZ coordinates as N x 3 array
    :rtype: numpy.ndarray
    """
    xyz = np.empty([len(df), 3])

    xyz[:, 0] = np.array(df.x)
    xyz[:, 1] = np.array(df.y)
    xyz[:, 2] = np.array(df.z)

    return xyz


def get_coordinates_of_conformer(mol):
    """Reads the coordinates of a conformer.

    :params mol: Molecule in RDKit format.
    :type mol: rdkit.Chem.rdchem.Mol

    :return: XYZ coordinates of molecule as N x 3 float array.
    :rtype: numpy.ndarray
    """

    symb = [a.GetSymbol() for a in mol.GetAtoms()]
    conf = mol.GetConformer()
    xyz = np.empty([mol.GetNumAtoms(), 3])

    for ia, name in enumerate(symb):
        position = conf.GetAtomPosition(ia)
        xyz[ia] = np.array([position.x, position.y, position.z])

    return xyz


def get_connectivity_matrix_from_mol(mol):
    """Calculates the binary bond connectivity matrix from a molecule.

    :param mol: Molecule in RDKit format.
    :type mol: rdkit.Chem.rdchem.Mol

    :return: Binary connectivity matrix (N x N) containing all molecular bonds.
    :rtype: numpy.ndarray
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


def get_bonds_matrix_from_mol(mol):
    """
    Calculates matrix of bond types from a molecule and returns as numpy array.
    Bond types are encoded as double:
     single bond (1.0)
     double bond (2.0)
     triple bond (3.0)
     aromatic bond (1.5).

    :param mol: Molecule in RDKit format.
    :type mol: rdkit.Chem.rdchem.Mol

    :return: Bond matrix (N x N) with bond types encoded as double.
    :rtype: numpy.ndarray

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


def get_bonds_list_from_mol(mol):
    """
    Calculates all bonds and bond types from a molecule and returns as dataframe.
    Bond types are encoded as double:
     single bond (1.0)
     double bond (2.0)
     triple bond (3.0)
     aromatic bond (1.5).

    :param mol: Molecule in RDKit format.
    :type mol: rdkit.Chem.rdchem.Mol

    :return: Bond information as dataframe with columns `atom1`, `atom2`, `type`.
    :rtype: pandas.DataFrame

    """
    bonds_list = []
    for b in mol.GetBonds():
        atom1 = b.GetBeginAtomIdx()
        atom2 = b.GetEndAtomIdx()
        btype = b.GetBondTypeAsDouble()
        bonds_list.append([atom1,atom2,btype])
    col = ['atom1','atom2','type']
    bonds_df = pd.DataFrame(bonds_list, columns=col)
    return bonds_df
