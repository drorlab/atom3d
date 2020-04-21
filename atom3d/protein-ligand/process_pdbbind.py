#!/usr/bin/env python
# coding: utf-8

import numpy as np
import os
import pandas as pd
import scipy.spatial
import sys
import argparse
sys.path.append('..')
from util import datatypes as dt
from rdkit import Chem
import Bio.PDB
from Bio.PDB.PDBIO import Select

def extract_binding_site(pdb_path):
    pdb_id = pdb_path.split('/')[-1]
    # get ligand coordinates
    lig_fname = pdb_id + '_ligand.mol2'
    lig_file = os.path.join(pdb_path, lig_fname)
    with open(lig_file) as f:
        lig_coords = get_ligand(f)

    # get protein coordinates
    prot_fname = pdb_id + '_protein.pdb'
    prot_file = os.path.join(pdb_path, prot_fname)
    with open(prot_file) as f:
        prot_coords = get_protein(f)

def get_ligand(ligfile):
    lig=Chem.SDMolSupplier(ligfile)[0]
    # if SDF fails, try Mol2
    if lig is None:
        print('trying mol2...')
        lig=Chem.MolFromMol2File(ligfile[:-4] + '.mol2')
    if lig is None:
        print('failed')
        return None
#     lig = Chem.MolFromMol2File(ligfile)
    lig = Chem.RemoveHs(lig)
    return lig


def get_protein(protfile):
    prot = dt.read_pdb(protfile)
    return prot


def get_pocket_res(protein, ligand, dist):
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
    def __init__(self, reslist):
        self.reslist = reslist
    def accept_residue(self, residue):
        if residue in self.reslist:
            return True
        else:
            return False


def process_files(pdb_path):
    pdb_id = pdb_path.split('/')[-1]
    # get ligand coordinates
    lig_fname = pdb_id + '_ligand.sdf'
    lig_file = os.path.join(pdb_path, lig_fname)
    lig = get_ligand(lig_file)

    # get protein coordinates
    prot_fname = pdb_id + '_protein.pdb'
    prot_file = os.path.join(pdb_path, prot_fname)
    prot = get_protein(prot_file)
    
    return prot, lig


def write_files(protein, ligand, pocket, out_path):
    pdbid = protein.id[:4]
    out_path = os.path.join(out_path, pdbid)
    if not os.path.exists(out_path):
        os.mkdir(out_path)
    # write protein to mmCIF file
    io = Bio.PDB.MMCIFIO()
    io.set_structure(protein)
    io.save(os.path.join(out_path, f"{pdbid}_protein.cif"))
    
    # write pocket to mmCIF file
    io.save(os.path.join(out_path, f"{pdbid}_pocket.cif"), PocketSelect(pocket))
    
    # write ligand to file
    writer = Chem.SDWriter(os.path.join(out_path, f"{pdbid}_ligand.sdf"))
    writer.write(ligand)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('datapath', type=str, help='directory where PDBBind is located')
    parser.add_argument('--dist', type=float, default=6.0, help='distance cutoff for defining pocket')
    args = parser.parse_args()
    
    if not os.path.exists(args.datapath):
        raise Exception('Path not found. Please enter valid path to PDBBind dataset.')
    
    out_path = os.path.join(args.datapath, 'cleaned')
    if not os.path.exists(out_path):
        os.mkdir(out_path)
    
    labels = pd.read_csv(os.path.join(args.datapath, 'pdbbind_refined_set_labels.csv'))
    valid_pdbs = labels.pdb.tolist()
        
    failed = 0
    for d in valid_pdbs:
        print(d)
        protein, ligand = process_files(os.path.join(args.datapath, 'refined-set', d))
        if ligand is None:
            failed += 1
            continue
        pocket = get_pocket_res(protein, ligand, args.dist)
        write_files(protein, ligand, pocket, out_path)
    print(failed, 'failed structures') # 135
    
    
if __name__ == "__main__":
    main()

