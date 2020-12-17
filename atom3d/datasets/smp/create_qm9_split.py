import argparse

import numpy as np

import atom3d.splits.splits as splits


def generate_split(excl_uncharacterized=True, excl_rdkitfails=True, out_dir_name='.', seed=42):

    num_molecules = 133885    

    # Load the list of molecules to ignore
    if excl_uncharacterized and not excl_rdkitfails: 
        unc_file = '../../data/qm9/raw/uncharacterized.txt'
        with open(unc_file, 'r') as f:
            exclude = [int(x.split()[0]) for x in f.read().split('\n')[9:-2]]
        assert len(exclude) == 3054 
    elif excl_uncharacterized and excl_rdkitfails:
        exclude = np.loadtxt('../../data/qm9/splits/excl.dat',dtype=int).tolist()
    elif excl_rdkitfails and not excl_uncharacterized:
        print('Excluding only RDKit fails is not implemented.')
        return
    else:
        exclude = []

    # Define indices to split the data set
    test_indices, vali_indices, train_indices = splits.random_split(num_molecules, vali_split=0.1, test_split=0.1, random_seed=seed, exclude=exclude)
    print('Training: %i molecules. Validation: %i molecules. Test: %i molecules.'%(len(train_indices),len(vali_indices),len(test_indices)))
    
    # Save the indices for the split
    np.savetxt(out_dir_name+'/indices_test.dat', np.sort(test_indices), fmt='%1d')
    np.savetxt(out_dir_name+'/indices_valid.dat',np.sort(vali_indices), fmt='%1d')
    np.savetxt(out_dir_name+'/indices_train.dat',np.sort(train_indices),fmt='%1d')
        
    return


############
# - MAIN - #

if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument("--excl_uncharacterized", type=bool, default=True)
    parser.add_argument("--excl_rdkitfails", type=bool, default=True)
    parser.add_argument("--out_dir_name", type=str, default='.' )
    args = parser.parse_args()
    
    generate_split(excl_uncharacterized = args.excl_uncharacterized, excl_rdkitfails = args.excl_rdkitfails, out_dir_name=args.out_dir_name)


