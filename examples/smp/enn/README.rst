Training an ENN model for SMP data
==================================


Installation
------------

To use the ENN model, make sure to install our Cormorant fork alongside ATOM3D as described in the `ENN installation instructions <https://atom3d.readthedocs.io/en/latest/training_models.html#enn>`_.


Dataset
-------


Download the randomly split SMP dataset from `the ATOM3D website <https://www.atom3d.ai/smp.html>`_.
Then extract it from the zipped archive.


Training
--------
  
The training scripts can be invoked from the example folder using::

    cd atom3d/examples/smp/enn
    python train.py --target mu --prefix smp-mu --load --datadir $LMDBDIR --format lmdb --num-epoch 50

where LMDBDIR is the path to the subfolder "/data" of the split LMDB dataset. All available targets are listed in the file `labels.txt <https://github.com/drorlab/atom3d/blob/master/examples/smp/enn/labels.txt>`_.

To see further options for training, use::

    python train.py --help
    

