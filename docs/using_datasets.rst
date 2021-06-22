Using ATOM3D datasets
=====================

All datasets in ATOM3D are provided in standardized LMDB format. LMDB allows for compressed, fast, random access to your structures, all within a single database. 

Downloading LMDB datasets
**************************

All datasets are hosted on Zenodo, and the links to download raw and split datasets in LMDB format can be found at `atom3d.ai <www.atom3d.ai>`_.
Alternatively, you can use the Python API:
    
.. code:: pycon

  >>> import atom3d.datasets as da
  >>> da.download_dataset('lba', TARGET_PATH, split=SPLIT_NAME)

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

By default, datasets contain all atoms in each molecular structure. However, many applications may require filtering the structure to remove undesirable atoms or focus on a specific region of the structure. 
This is easy to do by defining filter functions which operate on the ``atoms`` dataframe of a dataset item, returning a filtered version of the same dataframe. Several such filter functions are predefined in :mod:`atom3d.filters`.

For example, to remove non-standard residues from all proteins in a dataset:

.. code:: python

  from atom3d.filters import filters
  from atom3d.data.example import load_example_dataset

  dataset = load_example_dataset()
  for struct in dataset:
    struct['atoms'] = filters.standard_residue_filter(struct['atoms'])


It is also possible to combine multiple filters with :func:`atom3d.filters.filters.compose`. For example, to use only the first chain of a protein *and* remove non-standard residues:

.. code:: pycon

    >>> from atom3d.data.example import load_example_dataset
    >>> from atom3d.filters import filters
    >>> dataset = load_example_dataset()
    >>> struct = dataset[0]
    >>> filter1 = filters.single_chain_filter
    >>> filter2 = filters.standard_residue_filter
    >>> filter_fn = filters.compose(filter1, filter2)
    >>> struct['atoms'].shape
    (5220, 20)
    >>> struct['atoms'] = filter_fn(struct['atoms'])
    >>> struct['atoms'].shape
    (2568, 20)


These functions can also be readily extended by defining wrappers that return a filter function for a particular application. 
For example, the function :func:`atom3d.filters.sequence.form_seq_filter_against` creates a filter function that removes structures with greater than some sequence identity to any structure in a specified dataset (e.g. to filter train examples that are too similar to the test set).
  
.. code:: python

    from atom3d.filters.sequence import form_seq_filter_against
    from atom3d.datasets.datasets import LMDBDataset
    
    train_dataset = LMDBDataset(TRAIN_PATH)
    test_dataset = LMDBDataset(TEST_PATH)

    filter_fn = form_seq_filter_against(test_dataset, 0.3)

    for struct in train_dataset:
      struct['atoms'] = filter_fn(struct['atoms']) # returns empty dataframe if a match is found in test set


To automatically apply a filter to a dataset on the fly as each example is loaded, you can convert it to a transform function and pass it to any Dataset using the ``transform`` argument.

.. code:: python

    from atom3d.filters import filters
    from atom3d.datasets.datasets import LMDBDataset

    filter_fn = filters.standard_residue_filter
    transform_fn = filters.filter_to_transform(filter_fn) # convert filter function to transform function
    dataset = LMDBDataset(PATH, transform=transform_fn) # load dataset and apply transform

.. _splitting:

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

  * **Split randomly**

    The simplest splitting method is to split the dataset at random. 

  * **Split by sequence identity (proteins)**

  * **Split by scaffold (small molecules)**

  * **Split by year**


Defining your own splitting criteria
------------------------------------

  * **Split by cluster/group membership**

  * **Split by cluster/group size**

.. _examples:

Examples
********

The following examples illustrate some useful functionalities of ATOM3D using a small mock dataset.

  >>> from atom3d.data.example import load_example_dataset
  >>> dataset = load_example_dataset()

1. **Get coordinates of all atoms in a structure.**

.. code:: pycon

>>> import atom3d.util.formats as fo
>>> struct = dataset[0] # get first structure in dataset
>>> atoms_df = struct['atoms'] # load atom data for structure
>>> coords = fo.get_coordinates_from_df(atoms_df)
>>> coords.shape
(2568, 3)

2. **Get protein sequences from a structure.**

.. code:: pycon

>>> import atom3d.protein.sequence as seq
>>> struct = dataset[0] # get first structure in dataset
>>> atoms_df = struct['atoms'] # load atom data for structure
>>> chain_sequences = seq.get_chain_sequences(atoms_df) 
>>> chain_sequences # Contains sequences for all chains/monomers, identified by tuple of (ensemble, subunit, structure, model, chain)
[(('11as.pdb', '0', '11as.pdb', '1', 'A'), 'AYIAKQRQISFVKS...PAAVRESVPSLLN')]

3. **Extract all atoms within 5 Angstroms of a ligand**

In this example, the ligand is assumed to be stored as a subunit in the atoms dataframe, under the label "LIG". 
In practice, the ligand could be stored in different way (e.g. in a separate dataframe or under a different label), depending on how the dataset was constructed.

.. code:: pycon

>>> from atom3d.filters.filters import distance_filter
>>> import atom3d.util.formats as fo
>>> struct = dataset[0] # get first structure in dataset
>>> atoms_df = struct['atoms'] # load atom data for structure
>>> lig_coords = fo.get_coordinates_from_df(atoms_df[atoms_df['subunit']=='LIG']) # get coords of ligand
>>> df_filtered = distance_filter(atoms_df, lig_coords, dist=5.0)
