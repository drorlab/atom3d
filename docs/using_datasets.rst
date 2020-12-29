Using ATOM3D datasets
=====================

All datasets in ATOM3D are provided in standardized LMDB format. LMDB allows for compressed, fast, random access to your structures, all within a single database. 

Downloading LMDB datasets
*********************

All datasets can be downloaded in LMDB format from `atom3d.ai <atom3d.ai>`_, or using the Python API:
     
     .. code:: pycon
   
        >>> import atom3d.datasets as ds
        >>> ds.download_dataset('lba', '/path/to/target')

See the ATOM3D website or the :ref:`FAQ <sec:datasets-faq>` for more information about the datasets available.

Loading LMDB datasets in Python
********************************

After downloading or :doc:`creating </creating_datasets>` an LMDB dataset, it is easy to load it into Python. 
The resulting object acts just like a PyTorch Dataset, with a length and entries that can be accessed or iterated on.
See :doc:`/data_formats` for details on the format of each data entry.

.. code:: python

    from atom3d.datasets import LMDBDataset

    dataset = LMDBDataset(PATH_TO_LMDB)
    print(len(dataset))  # Print length
    print(dataset[0])  # Print 1st entry


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