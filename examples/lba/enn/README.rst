Training an ENN model for LBA data
==================================


Installation
------------


Create the environment
````````````````````````

Create a conda environment, defining the correct version of the CUDA toolkit (here: 10.1)::

    conda create --name cormorant python=3.7 pip scipy pytorch cudatoolkit=10.1 -c pytorch
    conda activate cormorant

If you do not know your CUDA version, you can find out via::

    nvcc --version
    
    
Install ATOM3D
````````````````````

Within the created environment, execute::

    pip install atom3d
    
    
OR install the development version from the ATOM3D repo::

   cd ~
   git clone https://github.com/drorlab/atom3d.git
   cd atom3d
   pip install -e .


Install Cormorant
````````````````````

The Cormorant fork used for this project can be cloned directly from the git repo using::

    cd ~
    git clone https://github.com/drorlab/cormorant.git


You can currently only install it in development mode by going to the directory with setup.py and running::

    cd cormorant
    python setup.py develop
    cd ~


Dataset
-------


Download the LBA dataset from `the ATOM3D website <https://www.atom3d.ai/lba.html>`_.
We recommend using the split based on 30% sequence identity but also provide a split based on 60% sequence identity as has been used by some groups in previous work.
Once the download has finished, extract the datasets from the zipped archive.


Training
--------
  
The training scripts can be invoked from the example folder using::

    cd atom3d/examples/lba/enn
    python train.py --target neglog_aff --load \
                    --prefix lba-id30_cutoff-06_maxnumat-600 \
                    --datadir $LMDBDIR --format lmdb \
                    --radius CUTOFF --maxnum MAXNUM \
                    --batch-size 1 --num-epoch 150

where LMDBDIR is the path to the subfolder "/data" of the split LMDB dataset.

To see further options for training, use::

    python train.py --help
    

