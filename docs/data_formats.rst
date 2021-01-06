ATOM3D data formats
===================

The Dataset object
************************

ATOM3D uses a standardized data format for 3D molecular structures, designed to maximize flexibility and efficiency without losing important structural information. 
The main way to interact with these datasets is through the Dataset objects in :mod:`atom3d.datasets.datasets`. These are essentially PyTorch datasets in which each item consists of several key-data pairs.

Each item (molecular structure) in an ATOM3D dataset contains at minimum the following keys:

* ``id`` (str): unique identifier for structure
* ``file_path`` (str): path to file from which structure was derived
* ``atoms`` (DataFrame): data describing 3D coordinates and atomic information

Depending on the dataset, other important data may be provided under additional keys:

* ``bonds`` (DataFrame): bond data is provided for small molecule structures
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
* ``atom2``: index of the second atom in the bond (corresponding to the ``serial_number`` field in the ``atoms`` dataframe)
* ``type``: bond type, encoded as Double (single bond = 1.0, double bond = 2.0, triple bond = 3.0, aromatic bond = 1.5).

