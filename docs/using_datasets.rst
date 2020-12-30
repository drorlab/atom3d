Using ATOM3D datasets
=====================

All datasets in ATOM3D are provided in standardized LMDB format. LMDB allows for compressed, fast, random access to your structures, all within a single database. 

Downloading LMDB datasets
*********************

All datasets can be downloaded in LMDB format from `atom3d.ai <atom3d.ai>`_, or using the Python API:
     
  .. code:: pycon

    >>> import atom3d.datasets as ds
    >>> ds.download_dataset('lba', TARGET_PATH)

See the ATOM3D website or the :ref:`FAQ <sec:datasets-faq>` for more information about the datasets available.

Loading datasets in Python
********************************

After downloading or :doc:`creating </creating_datasets>` an LMDB dataset, it is easy to load it into Python using ATOM3D. 
You can also load a dataset from a directory of files in any supported structural data format (currently PDB, SDF, XYZ, and Rosetta silent files).
The resulting object acts just like a PyTorch Dataset, with a length and entries that can be accessed or iterated on.
See :doc:`/data_formats` for details on the format of each data entry.

.. code:: python

    import atom3d.datasets as da

    dataset = da.load_dataset(PATH_TO_INPUT_DIR, {'lmdb', 'pdb','silent','sdf','xyz','xyz-gdb'})
    print(len(dataset))  # Print length 
    print(dataset[0].keys()) # Print keys stored in first structure


Filtering datasets
***********************


Splitting datasets
***********************

For most machine learning applications, the datasets will need to be split into train/validation/test subsets. 
Because the desired splitting methodology varies depending on the molecule type and the application, the standard way to split datasets in ATOM3D is by using pre-computed sets of indices into the dataset. 
These indices can be computed arbitrarily using any splitting function that takes in a dataset and returns indices to include in the train, validation, and test sets.

The :func:`atom3d.splits.splits.split` function then takes in a dataset and the split indices and returns the three corresponding sub-datasets (in the same format as the original dataset):

.. code:: python

    import atom3d.splits.splits as spl

    train_dataset, val_dataset, test_dataset = spl.split(dataset, indices_train, indices_val, indices_test)


Using standard splitting criteria
---------------------------------

ATOM3D provides splitting functions for many commonly used splitting methodologies in the :mod:`atom3d.splits.splits` module.

  * Split randomly

  * Split by sequence identity (proteins)

  * Split by scaffold (small molecules)

  * Split by year


Defining your own splitting criteria
------------------------------------

  * Split by cluster/group membership

  * Split by cluster/group size


Examples
********

#. **Get coordinates of all atoms in a structure.**

  .. code:: pycon

  >>> from atom3d.data.example import load_example_dataset
  >>> dataset = load_example_dataset()
  >>> struct = dataset[0] # get first structure in dataset
  >>> atoms_df = struct['atoms'] # load atom data for structure
  >>> coords = fo.get_coordinates_from_df(atoms_df)
  >>> coords.shape
  (100, 3)