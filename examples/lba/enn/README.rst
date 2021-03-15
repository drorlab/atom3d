Training an ENN model for LBA data
==================================


Installation
------------

To use the ENN model, make sure to install our Cormorant fork alongside ATOM3D as described in the `ENN installation documentation <https://atom3d.readthedocs.io/en/latest/training_models.html#enn>`_.


Dataset
-------


Download one of the split LBA dataset from `the ATOM3D website <https://www.atom3d.ai/lba.html>`_.
We recommend using the split based on 30% sequence identity but also provide a split based on 60% sequence identity as has been used by some groups in previous work.
Once the download has finished, extract the datasets from the zipped archive.


Training
--------
  
The training script can be invoked from the example folder using, e.g.::

    cd atom3d/examples/lba/enn
    python train.py --target neglog_aff --load \
                    --prefix lba-id30_cutoff-06_maxnumat-600 \
                    --datadir $LMDBDIR --format lmdb \
                    --radius 6 --maxnum 600 \
                    --batch-size 1 --num-epoch 150

where LMDBDIR is the path to the subfolder "/data" of the split LMDB dataset.

To see further options for training, use::

    python train.py --help
 
 
Analysis
--------

We will make standardized model evaluation code available soon.

