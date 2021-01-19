Using Cormorant
===============


Installation
------------

Creating the environment
````````````````````````

Create a conda environment with ATOM3D::

    conda create --name cormorant python=3.7 pip scipy pytorch cudatoolkit=10.1 -c pytorch
    conda activate cormorant
    pip install atom3d


Installing Cormorant
````````````````````

The Cormorant fork used for this project can be cloned directly from the git repo using::

    git clone https://github.com/drorlab/cormorant.git


You can currently install it in development mode by going to the directory with setup.py and running::

    cd cormorant
    python setup.py develop


Training
----------------

For Cormorant to read the data sets, they must be preprocessed (make sure to adapt file paths in the bash script)::

    bash prepare_npz_smp.sh
    
The training scripts can be invoked using::

    python train_smp.py


