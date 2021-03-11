Training an ENN model for SMP data
==================================


Installation
------------


Create the environment
````````````````````````

Create a conda environment, defining the correct version of the CUDA toolkit (here: 10.1)::

    conda create --name cormorant python=3.7 pip scipy pytorch cudatoolkit=10.1 -c pytorch
    conda activate cormorant

If you do not know your CUDA version, you can find out via

    nvcc --version
    
    
Install ATOM3D
````````````````````

Within the created environment, execute::

    pip install atom3d
    
    
OR install the development version from the atom3d repo::

   git clone https://github.com/drorlab/cormorant.git
   cd atom3d
   pip install -e .


Install Cormorant
````````````````````

The Cormorant fork used for this project can be cloned directly from the git repo using::

    git clone https://github.com/drorlab/cormorant.git


You can currently only install it in development mode by going to the directory with setup.py and running::

    cd cormorant
    python setup.py develop


Dataset
-------


Download the randomly split SMP dataset from `the ATOM3D website <https://www.atom3d.ai/smp.html>`_.
Then extract it from the zipped archive.


Training
--------
  
The training scripts can be invoked using::

    python train.py --target mu --prefix smp-mu --load --datadir $LMDBDIR --format lmdb --num-epoch 150

where LMDBDIR is the path to the subfolder data of the LMDB dataset.


