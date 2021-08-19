Training a 3D-CNN model for LBA data
====================================


Installation
------------

To use the three-dimensional convolutional neural network model (3D-CNN), no additional software besides ATOM3D and its requirements are necessary.


Dataset
-------

Download one of the *split* LBA datasets from `the ATOM3D website <https://www.atom3d.ai/lba.html>`_.
We recommend using the split based on 30% sequence identity but also provide a split based on 60% sequence identity as has been used by some groups in previous work.
Once the download has finished, extract the datasets from the zipped archive.


Training
--------
  
To start training, you need to define the directories LBA_DATA, OUTPUT_DIR, and LOG_DIR in which the data are stored, the output is saved, and the log files are saved, respectively. For example::

    export LBA_DATA=~/atom3d-data/lba/split-by-sequence-identity-30/data/
    export OUTPUT_DIR=~/atom3d/examples/lba/cnn3d/out/
    export LOG_DIR=~/atom3d/examples/lba/cnn3d/log/

The training script can then be invoked from the example folder::

    cd atom3d/examples/lba/enn
    python train.py 

To see further options for training, use::

    python train.py --help
 
