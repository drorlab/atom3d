# Example C++ Reader for ATOM3D data

## TO DO

* Read a single HDF5 file as in 'readdata.cpp', adapted to ATOM3D data.
* Check whether 'chunks.cpp' can be adapted for sharded datasets.

## Installation

Install the HDF library in serial mode:
```
sudo apt-get install libhdf5-serial-dev
```
The MPI mode does not have C++ support (only C or Fortran)

Compile by running
```
cmake .
make
```

Then execute
```
./readhdf
```

## Info

The code is derived from examples provided in the hdf5 repository.
It can be cloned via:
```
git clone https://bitbucket.hdfgroup.org/scm/hdffv/hdf5.git
```
and the examples are in the folder 'hdf5/c++/examples'


