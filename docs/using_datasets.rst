Using ATOM3D datasets
=====================

All datasets in ATOM3D are provided in standardized LMDB format. LMDB allows for compressed, fast, random access to your structures, all within a single database. 

Downloading LMDB datasets
*********************

All datasets can be downloaded in LMDB format from `atom3d.ai <atom3d.ai>`_, or using the Python API:
     
     .. code:: pycon
   
        >>> import atom3d.datasets as ds
        >>> ds.download_dataset('lba', '/path/to/target')

See the ATOM3D website or the `FAQ <datasets target>`_ for more information about the datasets available.

Loading LMDB datasets in Python.
********************************

After downloading or :doc:`creating </creating_datasets>` an LMDB dataset, it is easy to load it into Python.

.. code:: python

    from atom3d.datasets import LMDBDataset

    dataset = LMDBDataset(PATH_TO_LMDB)
    print(len(dataset))  # Print length
    print(dataset[0])  # Print 1st entry

ATOM3D data format
***********************

ATOM3D uses a standardized key-based LMDB data format for 3D molecular structures, designed to maximize flexibility and efficiency without losing important structural information.

Each structure in an ATOM3D dataset contains at minimum the following keys:

* ``id``: unique identifier for structure
* ``file_path``: path to file from which structure was derived
* ``atoms``: data describing 3D atom data, as DataFrame.

Depending on the dataset, other important data may be provided under additional keys:

* ``bonds``: bond data is provided for small molecule structures, in DataFrame format
* ``scores``/``labels``: task-specific scores or labels for each structure

The ``atoms`` dataframe
-----------------------

The ``atoms`` dataframe contains the bulk of the data for a 3D structure. The dataframe is constructed in a hierarchical manner to enable consistent data processing across datasets.

The first column is the ``ensemble``, which represents the highest level of structure, for example the PDB entry for a protein. 

The second column, the ``subunit``, is the subset of the ensemble representing the specific units of structure for the task/dataset. For example, for the protein structure ranking (PSR) task, each candidate structure for a given target protein would be assigned a unique subunit (here, the ensemble is the target protein).

The remainder of the columns contain the structural information. 
These follow the convention of the PDB file format: ``model`` - ``chain`` - ``hetero`` - ``insertion_code`` - ``residue`` - ``segid`` - ``resname`` - ``altloc`` - ``occupancy`` - ``bfactor`` - ``x`` - ``y`` - ``z`` - ``element`` - ``name`` - ``fullname`` - ``serial_number``.

For small molecules and nucleic acids, many of these columns are unused.

The ``bonds`` dataframe
-----------------------

The ``bonds`` dataframe contains the covalent bonding information for a molecular structure, especially for small molecules.
Each row of the dataframe represents a bond and contains three columns:

* ``atom1``: index of the first atom in the bond (corresponding to the ``serial_number`` field in the ``atoms`` dataframe)
* ``atom2``: index of the second atom in the bond (corresponding to ``serial_number`` field in the ``atoms`` dataframe)
* ``type``: bond type, encoded as Double (single bond = 1.0, double bond = 2.0, triple bond = 3.0, aromatic bond =1.5).

Filtering and splitting
***********************

Using default splitting criteria
--------------------------------

TODO

Defining your own splitting criteria
------------------------------------

TODO

Examples
********

#. **Get coordinates of all atoms in a structure.**

  .. code:: pycon

  >>> import atom3d.util.formats as fo
  >>> from atom3d.examples import example_df
  >>> coords = fo.get_coordinates_from_df(example_df)
  >>> coords.shape
  (100, 3)