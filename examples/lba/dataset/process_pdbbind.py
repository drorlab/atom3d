#!/usr/bin/env python
# coding: utf-8


import os

import scipy.spatial

import atom3d.util.file as fi
import atom3d.util.formats as ft
import get_labels as lab

from rdkit import Chem
import Bio.PDB
from Bio.PDB.PDBIO import Select
from tqdm import tqdm
import argparse



def get_ligand(ligfile):
    """
    Read ligand from PDB dataset into RDKit Mol.
    :param ligfile: file containing ligand information in SDF format
    :type ligfile: str
    """
    lig=Chem.SDMolSupplier(ligfile)[0]
    # Many SDF in PDBBind do not parse correctly. If SDF fails, try loading the mol2 file instead
    if lig is None:
        print('trying mol2...')
        lig=Chem.MolFromMol2File(ligfile[:-4] + '.mol2')
    if lig is None:
        print('failed')
        return None
    lig = Chem.RemoveHs(lig)
    return lig


def get_pocket_res(protein, ligand, dist):
    """
    Given a co-crystallized protein and ligand, extract residues within specified distance of ligand.

    :param protein: Biopython object containing receptor protein
    :type protein: Bio.PDB.Structure
    :param ligand: RDKit molecule object containing co-crystallized ligand
    :type ligand: rdkit.Chem.rdchem.Mol
    :param dist: distance cutoff for defining binding pocket, in Angstrom
    :type dist: float

    :return key_residues: key binding site residues
    :rtype key_residues:  set containing Bio.PDB.Residue objects
    """
    # get protein coordinates
    prot_atoms = [a for a in protein.get_atoms()]
    prot_coords = [atom.get_coord() for atom in prot_atoms]

    # get ligand coordinates
    lig_coords = []
    for i in range(0, ligand.GetNumAtoms()):
        pos = ligand.GetConformer().GetAtomPosition(i)
        lig_coords.append([pos.x, pos.y, pos.z])

    kd_tree = scipy.spatial.KDTree(prot_coords)
    key_pts = kd_tree.query_ball_point(lig_coords, r=dist, p=2.0)
    key_pts = set([k for l in key_pts for k in l])

    key_residues = set()
    for i in key_pts:
        atom = prot_atoms[i]
        res = atom.get_parent()
        if res.get_resname() == 'HOH':
            continue
        key_residues.add(res)
    return key_residues


class PocketSelect(Select):
    """
    Selection class for subsetting protein to key binding residues. This is a subclass of :class:`Bio.PDB.PDBIO.Select`.
    """
    def __init__(self, reslist):
        self.reslist = reslist
    def accept_residue(self, residue):
        if residue in self.reslist:
            return True
        else:
            return False


def process_files(input_dir):
    """
    Process all protein (pdb) and ligand (sdf) files in input directory.

    :param input dir: directory containing PDBBind data
    :type input_dir: str

    :return structure_dict: dictionary containing each structure, keyed by PDB code. Each PDB is a dict containing protein as Biopython object and ligand as RDKit Mol object
    :rtype structure_dict: dict
    """
    structure_dict = {}
    pdb_files = fi.find_files(input_dir, 'pdb')

    for f in tqdm(pdb_files, desc='pdb files'):
        pdb_id = fi.get_pdb_code(f)
        if pdb_id not in structure_dict:
            structure_dict[pdb_id] = {}
        if '_protein' in f:
            prot = ft.read_any(f)
            structure_dict[pdb_id]['protein'] = prot

    lig_files = fi.find_files(input_dir, 'sdf')
    for f in tqdm(lig_files, desc='ligand files'):
        pdb_id = fi.get_pdb_code(f)
        structure_dict[pdb_id]['ligand'] = get_ligand(f)

    return structure_dict



def write_files(pdbid, protein, ligand, pocket, out_path):
    """
    Writes cleaned structure files for protein, ligand, and pocket.

    :param pdbid: PDB ID for protein
    :type pdbid: str
    :param protein: object containing proteins
    :type ligand: object containing ligand molecules
    :param pocket: residues contained in pocket, as output by get_pocket_res()
    :type pocket: set containing Bio.PDB.Residue objects
    :param out_path: directory to write files to
    :type out_path: str
    """
    # write protein to mmCIF file
    io = Bio.PDB.MMCIFIO()
    io.set_structure(protein)
    io.save(os.path.join(out_path, f"{pdbid}_protein.cif"))

    # write pocket to mmCIF file
    io.save(os.path.join(out_path, f"{pdbid}_pocket.cif"), PocketSelect(pocket))

    # write ligand to file
    writer = Chem.SDWriter(os.path.join(out_path, f"{pdbid}_ligand.sdf"))
    writer.write(ligand)


def produce_cleaned_dataset(structure_dict, out_path, dist=6.0):
    """
    Generate and save cleaned dataset, given dictionary of structures processed by process_files.

    :param structure_dict: dictionary containing protein and ligand structures processed from input files. Should be a nested dict with PDB ID in first level and keys 'protein', 'ligand' in second level.
    :type structure_dict: dict
    :param out_path: path to output directory
    :type out_path: str
    :param dist: distance cutoff for defining binding pocket (in Angstrom), default is 6.0
    :type dist: float, optional
    """
    proteins = []
    pockets = []
    for pdb, data in tqdm(structure_dict.items(), desc='writing to files'):
        protein = structure_dict[pdb]['protein']
        ligand = structure_dict[pdb]['ligand']
        # check for failed ligand (due to bad structure file)
        if ligand is None:
            continue
        pocket_res = get_pocket_res(protein, ligand, dist)
        write_files(pdb, protein, ligand, pocket_res, out_path)

def generate_labels(data_dir, out_dir):
    lab.main(data_dir, out_dir)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('data_dir', type=str, help='directory where PDBBind is located')
    parser.add_argument('--dist', type=float, default=6.0, help='distance cutoff for defining pocket')
    parser.add_argument('--out_dir', type=str, default=os.getcwd(), help='directory to place cleaned dataset')
    args = parser.parse_args()

    if not os.path.exists(args.data_dir):
        raise Exception('Path not found. Please enter valid path to PDBBind dataset.')

    if not os.path.exists(args.out_dir):
        os.mkdir(args.out_dir)

    structures = process_files(args.data_dir)
    produce_cleaned_dataset(structures, args.out_dir, args.dist)
    generate_labels(args.data_dir, args.out_dir)


if __name__ == "__main__":
    main()
