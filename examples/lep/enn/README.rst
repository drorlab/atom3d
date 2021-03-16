Training an ENN model for LEP data
==================================


Installation
------------

To use the ENN model, make sure to install our Cormorant fork alongside ATOM3D as described in the `ENN installation instructions <https://atom3d.readthedocs.io/en/latest/training_models.html#enn>`_.


Dataset
-------


Download the *split* LEP dataset from `the ATOM3D website <https://www.atom3d.ai/lba.html>`_.
Once the download has finished, extract the dataset from the zipped archive.


Training
--------
  
The training script can be invoked from the example folder using, e.g.::

    cd atom3d/examples/lep/enn

    python train.py --prefix lep_cutoff-08_maxnum-400-noH \
                    --datadir $LMDBDIR --format LMDB\
                    --drop --radius 08 --maxnum 400 \
                    --batch-size 1 --num-epoch 50 \
                    --load

where LMDBDIR is the path to the subfolder "/data" of the split LMDB dataset.

To see further options for training, use::

    python train.py --help
 
 
Analysis
--------

We will make standardized model evaluation code available soon.

