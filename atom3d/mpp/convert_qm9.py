import Bio.PDB
import numpy as np
import pandas as pd
import sys, os, glob, csv
sys.path.append('../..')
import atom3d.util.datatypes as dt

# -------------------
# -- Header legend --
# -------------------
#
#   1  tag     -         ‘gdb9’ string to facilitate extraction
#   2  i       -         Consecutive, 1-based integer identifier
#   3  A      GHz        Rotational constant
#   4  B      GHz        Rotational constant
#   5  C      GHz        Rotational constant
#   6  μ      D          Dipole moment
#   7  α      a_0^3      Isotropic polarizability
#   8  εHOMO  Ha         Energy of HOMO
#   9  εLUMO  Ha         Energy of LUMO
#  10  εgap   Ha         Gap (εLUMO−εHOMO)
#  11  <R^2>  a_0^2      Electronic spatial extent
#  12  zpve   Ha         Zero point vibrational energy
#  13  U0     Ha         Internal energy at 0 K
#  14  U      Ha         Internal energy at 298.15 K
#  15  H      Ha         Enthalpy at 298.15 K
#  16  G      Ha         Free energy at 298.15 K
#  17  Cv     cal/mol/K  Heat capacity at 298.15 K
#
# -------------------
# -- Footer legend --
# -------------------
#
#  Harmonic vibrational frequencies (3*na−5 or 3*na-6, in cm−1)
#  (na = number of atoms) 
#
# ---------------------
# -- Reference paper --
# ---------------------
#
#  Quantum chemistry structures and properties of 134 kilo molecules
#  Raghunathan Ramakrishnan, Pavlo O. Dral, Matthias Rupp & O. Anatole von Lilienfeld 
#  Scientific Data volume 1, Article number: 140022 (2014)
#  https://www.nature.com/articles/sdata201422
#

def convert_xyz_folder_to_mmcif(xyz_folder = 'qm9/qm9_original/dsgdb9nsd_xyz',
                                filename_pattern = 'dsgdb9nsd_%06i',
                                mmcif_folder = 'qm9/qm9_mmcif_csv',
                                indices = None, num_mol = 133885):
    
    # determine which indices to write
    if indices is None:
        indices = np.arange(num_mol)+1
    else:
        num_mol = len(indices)
        
    # Create the new folder (if does not exist)
    try:
        os.makedirs(mmcif_folder)
    except FileExistsError:
        pass
    
    
    # Names of the output files
    fn_scalars = mmcif_folder+'/scalar_quantities.csv'
    fn_frequ = mmcif_folder+'/harmonic_vibrational_frequencies.csv'

    # Open the output files
    with open(fn_scalars,'w') as file_scalars, open(fn_frequ,'w') as file_frequ:

        # Define the CSV writers
        wr_sc = csv.writer(file_scalars)
        wr_fr = csv.writer(file_frequ)

        # Iterate through all indices
        for i in indices:

            # Read data from the xyz file (provides a dictionary)
            data = dt.read_xyz(xyz_folder+'/'+filename_pattern%(i)+'.xyz')

            # Create lists from header and footer
            scalar_quantities = data['header'].split()[1:] # omit 'gdb' string at beginning
            harmon_vibr_frequ = data['footer'].split()

            # Write data with smiles strings to the data files
            wr_sc.writerow([data['smiles']] + [data['inchi']] + scalar_quantities)
            wr_fr.writerow([data['smiles']] + [data['inchi']] + harmon_vibr_frequ)

            # Obtain the structure from the xyz coordinates and write it to mmcif
            s = dt.bp_from_xyz_dict(data, struct_name = data['smiles']+' '+data['inchi'])
            dt.write_mmcif(mmcif_folder+'/'+filename_pattern%(i)+'.cif', s)
        
    return


if __name__ == "__main__":
    convert_xyz_folder_to_mmcif()
