Using Cormorant
===============


Preparing the data sets
-----------------------

For Cormorant to read the data sets, they must be preprocessed::

    bash prepare_qm9.sh

while still the atom3d environment should be activated.


Installation
------------

Creating the environment
````````````````````````

Create a separate conda environment::

    conda create --name cormorant python=3.7 pip scipy pytorch cudatoolkit=10.1 -c pytorch

Cloning the git repo
`````````````````````

Cormorant can be cloned directly from the git repo using::

    git clone https://github.com/risilab/cormorant.git

Installing Cormorant
````````````````````

You can currently install it in development mode by going to the directory with setup.py and running::

    python setup.py develop


Training example
----------------

The training scripts can be invoked using::

    python train_qm9.py


