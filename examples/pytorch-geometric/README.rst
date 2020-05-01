Using PyTorch Geometric
=======================

PyTorch Geometric is available at https://github.com/rusty1s/pytorch_geometric

Preparing the data sets
-----------------------

For PyTorch Geometric to read the data sets, they must be preprocessed::

    bash prepare_qm9.sh
    bash prepare_pdbbind.sh

while still the atom3d environment should be activated.


Installation
------------

Creating the environment
````````````````````````

Create a separate conda environment::

    conda create --name geometric -c pytorch -c rdkit pip rdkit pytorch=1.5 cudatoolkit=10.2


Installing PyTorch Geometric 
````````````````````````````

You can install it by running::

    conda activate geometric
    pip install torch-scatter==latest+${CUDA} -f https://pytorch-geometric.com/whl/torch-1.5.0.html
    pip install torch-sparse==latest+${CUDA} -f https://pytorch-geometric.com/whl/torch-1.5.0.html
    pip install torch-cluster==latest+${CUDA} -f https://pytorch-geometric.com/whl/torch-1.5.0.html
    pip install torch-spline-conv==latest+${CUDA} -f https://pytorch-geometric.com/whl/torch-1.5.0.html
    pip install torch-geometric


Training example
----------------

The training scripts can be invoked using::

    ???


