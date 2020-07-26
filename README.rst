ATOM3D: Tasks On Molecules in 3 Dimensions
==========================================

[![Documentation
Status](https://readthedocs.org/projects/atom3d/badge/?version=latest)](http://atom3d.readthedocs.io/?badge=latest)

ATOM3D enables machine learning on three dimensional molecular structure.

Usage
-----

To load a sharded dataset:

   import atom3d.shard.shard as sh

   # Load dataset split into fragments (or shards).
   sharded = sh.load_sharded('sharded/candidates/structures@21')

   # Iterate through shards.
   for shard_num in sharded.get_num_shards():
      structures = sharded.read_shard(shard_num)
      # You can also load associated metadata.
      labels = sharded.read_shard(shard_num, 'labels')

Features
--------

- Access to several dataset involving 3D molecular structure. 
- Sharded data format for storing large 3D molecular datasets.
- Utilities for splitting/filtering data based on many criteria.


Installation
------------

Install with:

    pip install -r requirements.txt
    
To use rdkit functionality, please install within conda:

    conda create -n atom3d python=3.6 pip rdkit
    conda activate atom3d
    pip install -r requirements.txt

Contribute
----------

- Issue Tracker: github.com/drorlab/atom3d/issues
- Source Code: github.com/drorlab/atom3d

Support
-------

If you are having issues, please let us know.
We have a mailing list located at: atom3d@googlegroups.com

License
-------

The project is licensed under the MIT license.
