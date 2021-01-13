Contributing to ATOM3D
======================

You can support the mission of ATOM3D in several different ways: by reporting bugs and flaws, by requesting new features that would benefit your work, by contributing a dataset, by contributing a new class of model, or by writing or improving code to handle them. 

Machine learning on three-dimensional molecular structure is only at its beginning. The datasets and methods here are only a snapshot of the current state of the field and we see ATOM3D as a platform to curate and share them with the community.  See below for the most common ways to help us in this endeavour.
Finally, we are always happy hear about your machine learning success stories.

Report a bug or request a feature
***********************************

ATOM3D is open-source and available on `Github <https://github.com/drorlab/atom3d>`_. Please submit issues or requests using the `issue tracker <https://github.com/drorlab/atom3d/issues>`_.

Add a dataset
***********************************

We distribute datasets in a JSON-encoded LMDB format which is described in the  `corresponding documentation <https://atom3d.readthedocs.io/en/latest/data_formats.html>`_. The ATOM3D library provides many tools to prepare such datasets from commonly used datatypes in the field (PDB, SDF, etc.). Ideally, you would provide the code to prepare the datasets along with them. See the dataset-specific folders in :mod:`atom3d.datasets` for examples.

Once you have prepared the dataset, please contact us so we can add it to ATOM3D.

Add a model
***********************************

We are still figuring out the best way to organize the various models.

Please contact us if you have a model that you would like to be added to ATOM3D.

Add new functionality 
***********************************

We welcome any kind of contributions to improve or expand the ATOM3D code. In particular, we are interested in readers for new data formats and new ways to split, filter, or transform datasets. ATOM3D is maintained on `Github <https://github.com/drorlab/atom3d>`_ so you can fork it and create a pull request. Please make sure to properly test your contribution before the request. For large or complicated contributions, please get in contact so we can coordinate them with you. We explain two of the most common and straightforward cases (file readers and splits) below:

New file readers
-----------------------------------
To add the possibility to read a new type of datasets, add a class to the file :mod:`atom3d/datasets/datasets.py <atom3d.datasets.datasets>`. This class should inherit the PyTorch Dataset class and be compatible with the internal :doc:`ATOM3D data format <data_formats>`. Use one of the existing classes like :class:`LMDBDataset(Dataset) <atom3d.datasets.datasets.LMDBDataset>`, :class:`PDBDataset(Dataset) <atom3d.datasets.datasets.PDBDataset>` or :class:`SDFDataset(Dataset) <atom3d.datasets.datasets.SDFDataset>` as a template.

New splitting functions
-----------------------------------
To add a new way to split a dataset, you can build on the generic split functions in :mod:`atom3d.splits.splits`. For example:

 - The function :func:`split_by_group <atom3d.splits.splits.split_by_group>` assigns elements to splits based on pre-defined lists of values. A "value function", defining the value in the data entry to which the values in the list correspond, has to be provided as an argument.
 - Similarly, :func:`split_by_group_size <atom3d.splits.splits.split_by_group_size>` assigns elements so that the largest groups (i.e. most common examples) are in the training set and the smallest groups (i.e. less common examples) are in the test set.

You can create more specific split functions from them by defining the value function, as we do in the following examples (which are already part of ATOM3D):

     .. code:: python
   
        from functools import partial
        split_by_year = partial(split_by_group, value_fn=lambda x: x['year'])
        split_by_scaffold = partial(split_by_group_size, value_fn=lambda x: x['scaffold'])


