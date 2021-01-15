# ATOM3D: Tasks On Molecules in 3 Dimensions

[![Documentation
Status](https://readthedocs.org/projects/atom3d/badge/?version=latest)](http://atom3d.readthedocs.io/?badge=latest)
![Package](https://github.com/drorlab/atom3d/workflows/package/badge.svg)
[![codecov](https://codecov.io/gh/drorlab/atom3d/branch/master/graph/badge.svg?token=DHH78W45AB)](https://codecov.io/gh/drorlab/atom3d)
[![PyPI version](https://badge.fury.io/py/atom3d.svg)](https://badge.fury.io/py/atom3d)

[ATOM3D](https://www.atom3d.ai/) enables machine learning on three-dimensional molecular structure.

## Features

* Access to several datasets involving 3D molecular structure. 
* LMDB data format for storing lots of molecules (and associated metadata).
* Utilities for splitting/filtering data based on many criteria.

For more detailed information, [read the documentation](https://atom3d.readthedocs.io/en/latest/).

## Installation/Usage

Install with:

```
pip install atom3d
```
    
To use rdkit functionality, please install within conda:

```
conda create -n atom3d python=3.6 pip rdkit
conda activate atom3d
pip install atom3d
```

### LMDB datasets

LMDB allows for compressed, fast, random access to your structures, all within a
single database.  Currently, we support creating LMDB datasets from PDB files, silent files, and xyz files.

#### Creating an LMDB dataset

From command line:
```
python -m atom3d.datasets PATH_TO_PDB_DIR PATH_TO_LMDB_OUTPUT --filetype {pdb,silent,xyz,xyz-gdb} 
```

#### Loading an LMDB dataset

From python:
```
from atom3d.datasets import LMDBDataset

dataset = LMDBDataset(PATH_TO_LMDB)
print(len(dataset))  # Print length
print(dataset[0])  # Print 1st entry
```

From C++:

 We provide example C++ code to load the LMDB datasets [here](https://github.com/drorlab/atom3d/tree/master/atom3d/cpp/lmdb-reader).
 
For more usage, please see the [documentation](https://atom3d.readthedocs.io/en/latest/).

## Contribute

As a living repository, we welcome contributions of additional datasets, methods, and functionality!  See the [Contributing](https://atom3d.readthedocs.io/en/latest/contributing.html) section of the documentation for details.

## Support

For support, please file an issue at https://github.com/drorlab/atom3d/issues.

## License

The project is licensed under the MIT license.

## Reference

We provide an overview on ATOM3D and details on the preparation of all datasets in our preprint:

> R. J. L. Townshend, M. Vögele, P. Suriana, A. Derry, A. Powers, Y. Laloudakis, S. Balachandar, B. Anderson, S. Eismann, R. Kondor, R. B. Altman, R. O. Dror "ATOM3D: Tasks On Molecules in Three Dimensions", [arXiv:2012.04035](https://arxiv.org/abs/2012.04035)
  
Please cite this work if some of the ATOM3D code or datasets are helpful in your scientific endeavours. For specific datasets, please also cite the respective original source(s), given in the preprint.
