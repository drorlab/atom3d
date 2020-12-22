Creating new datasets
==========================

In addition to the eight pre-curated datasets, you can also create your own datasets in the same standardized LMDB format. Currently, we support creating LMDB datasets from PDB files, SDF files, silent files, and xyz files.

.. _creating ref:
Creating dataset from input files
***************************

Assuming a directory containing all of the files you wish to process, you can create a new LMDB dataset from the command line:

  .. code:: bash

    python -m atom3d.datasets PATH_TO_INPUT_DIR PATH_TO_LMDB_OUTPUT --filetype {pdb,silent,xyz,xyz-gdb} 
