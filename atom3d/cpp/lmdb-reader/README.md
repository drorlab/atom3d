# Example C++ Reader for ATOM3D data in LMDB format


## Installation under Linux

Install the LMDB and gzip libraries:
```
sudo apt install liblmdb-dev
sudo apt install zlib1g-dev
```

Install the JSON library via conda
```
conda activate atom3d
conda install -c conda-forge nlohmann_json
```

Compile and execute by running
```
make main
./main
```

