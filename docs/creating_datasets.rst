Creating new datasets
==========================

In addition to the eight pre-curated datasets, you can also create your own datasets in the same standardized LMDB format. Currently, we support creating LMDB datasets from a set of PDB files, SDF files, silent files, or xyz files.

Create a dataset from input files
***********************************

Assuming a directory containing all of the files you wish to process, you can create a new LMDB dataset from the command line:

  .. code:: bash

    python -m atom3d.datasets PATH_TO_INPUT_DIR PATH_TO_LMDB_OUTPUT --filetype {pdb,silent,xyz,xyz-gdb} 

You can also load the dataset first in Python before writing it to LMDB format using the :mod:`atom3d.datasets.datasets` module:

  .. code:: python

    import atom3d.datasets.datasets as da

    # Load dataset from directory of PDB files
    dataset = da.load_dataset(PATH_TO_INPUT_DIR, 'pdb')
    # Create LMDB dataset from PDB dataset
    da.make_lmdb_dataset(dataset, PATH_TO_LMDB_OUTPUT)
                         
                         
Modify a dataset (add labels etc.)
***********************************

To modify a dataset you can load it in Python and define the modification via the ``transform`` option. 
The most common modification is adding labels, which are usually provided separate from PDB or SDF files.
In the following example, we assume that they are saved in CSV files with the same names as the corresponding PDB files.

  .. code:: python

    import os
    import pandas as pd
    import atom3d.datasets.datasets as da

    def add_label(item):
        # Remove the file ending ".pdb" from the ID
        name = item['id'][:-4]
        # Get label data
        label_file = os.path.join(PATH_TO_LABELS_DIR, name+'.csv')
        # Add label data in form of a data frame
        item['label'] = pd.read_csv(label_file)
        return item
        
    # Load dataset from directory of PDB files
    dataset = da.load_dataset(PATH_TO_INPUT_DIR, 'pdb', transform=add_label)
    
    # Create LMDB dataset from PDB dataset
    da.make_lmdb_dataset(dataset, PATH_TO_LMDB_OUTPUT)

You can flexibly use the `transform` option to modify any aspect of a dataset. For example, if you want to shift all structures in x direction, use the following function:

  .. code:: python
  
    def my_transformation(item):
        item['atoms']['x'] += 3
        return item
      
      
Split a dataset
***********************************

Once you have processed your dataset, you probably want to split it in training, validation, and test sets. 
In the following example, we assume that we want to split the dataset generated above according to a predefined split and that the IDs for the structures that belong in each dataset are defined in the files *train.txt*, *valid.txt* and *test.txt*.

  .. code:: python
        
    import atom3d.splits.splits as spl
    
    # Load split values
    tr_values = pd.read_csv('train.txt',header=None)[0].tolist()
    va_values = pd.read_csv('valid.txt',header=None)[0].tolist()
    te_values = pd.read_csv('test.txt',header=None)[0].tolist()
    
    # Create splits
    split_ds = spl.split_by_group(dataset,
                                  value_fn = lambda x: x['id'],
                                  train_values = tr_values,
                                  val_values   = va_values,
                                  test_values  = te_values)
    
    # Create split LMDB datasets 
    for s, split_name in enumerate(['training','validation','test']):
        # Create the output directory if it does not exist yet
        split_dir = os.path.join(PATH_TO_LMDB_OUTPUT, split_name)
        os.makedirs(split_dir, exist_ok=True)
        # Create LMDB dataset for the current split
        da.make_lmdb_dataset(split_ds[s], split_dir)

There are many ways to split datasets and we provide functions for many of them in the the :mod:`atom3d.splits` module. They are described in more detail `here <https://atom3d.readthedocs.io/en/latest/using_datasets.html#splitting-datasets>`_.
