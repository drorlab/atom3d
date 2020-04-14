"""Methods to convert between different file formats."""
import collections as col
import gzip
import os

import Bio.PDB
import pandas as pd


def read_pdb(pdb_file):
    """Load pdb file in to biopython representation."""
    parser = Bio.PDB.PDBParser(QUIET=True)
    _, ext = os.path.splitext(pdb_file)
    name = os.path.basename(pdb_file)
    if ext == ".gz":
        bp = parser.get_structure(
            name, gzip.open(pdb_file, mode='rt', encoding='latin1'))
    elif ".pdb" in ext:
        bp = parser.get_structure(name, pdb_file)
    else:
        raise ValueError("Unrecognized filetype " + pdb_file)
    return bp


def read_mmcif(mmcif_file):
    """Load mmcif file in to biopython representation."""
    parser = Bio.PDB.MMCIFParser(QUIET=True)
    return parser.get_structure(os.path.basename(mmcif_file), mmcif_file)


def bp_to_df(bp):
    """Convert biopython representation to pandas dataframe representation."""
    df = col.defaultdict(list)
    for atom in Bio.PDB.Selection.unfold_entities(bp, 'A'):
        residue = atom.get_parent()
        chain = residue.get_parent()
        model = chain.get_parent()
        df['structure'].append(bp._id)
        df['model'].append(str(model.serial_num))
        df['chain'].append(residue.get_full_id()[2])
        df['residue'].append(str(atom.get_parent().get_id()[1]) +
                             atom.get_parent().get_id()[2])
        df['resname'].append(residue.get_resname())
        df['x'].append(atom.get_coord()[0])
        df['y'].append(atom.get_coord()[1])
        df['z'].append(atom.get_coord()[2])
        df['element'].append(atom.get_id()[0])
        df['atom_name'].append(atom.get_name())
        df['serial_number'].append(str(atom.serial_number))
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
    for (structure, s_atoms) in df.groupby(['structure']):
        new_structure = Bio.PDB.Structure.Structure(structure)
        for (model, m_atoms) in df.groupby(['model']):
            new_model = Bio.PDB.Model.Model(model)
            for (chain, c_atoms) in m_atoms.groupby(['chain']):
                new_chain = Bio.PDB.Chain.Chain(chain)
                for (residue, r_atoms) in c_atoms.groupby(['residue']):
                    resname = r_atoms['resname'].unique()
                    if len(resname) == 1:
                        raise RuntimeError(
                            'More than one resname in provided residue')
                    resname = resname[0]
                    new_residue = Bio.PDB.Residue.Residue(
                        (' ', int(residue[:-1]), residue[-1]), resname, '')
                    for row, atom in r_atoms.iterrows():
                        new_atom = Bio.PDB.Atom.Atom(
                            atom['atom_name'],
                            [atom['x'], atom['y'], atom['z']],
                            1,
                            1,
                            ' ',
                            atom['atom_name'],
                            atom['serial_number'],
                            atom['element'])
                        new_residue.add(new_atom)

                    new_chain.add(new_residue)
                new_model.add(new_chain)
            new_structure.add(new_model)
        all_structures.append(new_structure)
    return all_structures
