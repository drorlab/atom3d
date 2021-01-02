Creating new datasets
==========================

In addition to the eight pre-curated datasets, you can also create your own datasets in the same standardized LMDB format. Currently, we support creating LMDB datasets from a set of PDB files, SDF files, silent files, or xyz files.

Creating dataset from input files
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
    da.make_lmdb_dataset(dataset, PATH_TO_LMDB_OUTPUT,
                         filter_fn=None, serialization_format='json',
                         include_bonds=False)
                         
                         
Modifying a dataset (adding labels etc.)
***********************************

To modify a dataset you can load it in Python and define the modification via the :mod:`transform` option. 
The most common modification is adding labels, which are usually provided separate from PDB or SDF files.
In the following example, we assume that they are saved in CSV files with the same names as the corresponding PDB files.

  .. code:: python

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
    da.make_lmdb_dataset(dataset, PATH_TO_LMDB_OUTPUT,
                         filter_fn=None, serialization_format='json',
                         include_bonds=False)

You can flexibly use the :mod:`transform` option to modify any aspect of a dataset. For example, if you want to shift all structures in x direction, use the following function:

  .. code:: python
  
    def my_transformation(item):
        item['atoms']['x'] += 3
        return item
      
