Creating new datasets
==========================

In addition to the eight pre-curated datasets, you can also create your own datasets in the same standardized LMDB format. Currently, we support creating LMDB datasets from a set of PDB files, SDF files, silent files, or xyz files.

Creating dataset from input files
***********************************

Assuming a directory containing all of the files you wish to process, you can create a new LMDB dataset from the command line:

  .. code:: bash

    python -m atom3d.datasets PATH_TO_INPUT_DIR PATH_TO_LMDB_OUTPUT --filetype {pdb,silent,xyz,xyz-gdb} 

You can also load the dataset first in Python before writing it to LMDB format:

  .. code:: python

    import atom3d.datasets as da

    # Load dataset from directory of PDB files
    dataset = da.load_dataset(PATH_TO_INPUT_DIR, 'pdb')
    # Create LMDB dataset from PDB dataset
    da.make_lmdb_dataset(dataset, PATH_TO_LMDB_OUTPUT,
                         filter_fn=None, serialization_format='json',
                         include_bonds=False)