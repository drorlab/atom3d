Training a GNN model for LBA data
==================================


Installation
------------

The GNN models require Pytorch Geometric — see details in either `ATOM3D <https://atom3d.readthedocs.io/en/latest/training_models.html#model-specific-installation-instructions>`_ or `PTG <https://pytorch-geometric.readthedocs.io/en/latest/notes/installation.html>`_ documentation.

Dataset
-------


Download one of the *split* LBA datasets from `the ATOM3D website <https://www.atom3d.ai/lba.html>`_.
We recommend using the split based on 30% sequence identity but also provide a split based on 60% sequence identity as has been used by some groups in previous work.
Once the download has finished, extract the datasets from the zipped archive.


Training
--------
  
The training script can be invoked from the example folder using, e.g.::

    cd atom3d/examples/lba/gnn
    python train.py --data_dir $LMDBDIR
                    --mode train
                    --batch_size 16
                    --num_epochs 50
                    --learning_rate 1e-4
                    
where LMDBDIR is the path to the subfolder "/data" of the split LMDB dataset.

To see further options for training, use::

    python train.py --help
 
 
Analysis
--------

We will make standardized model evaluation code available soon.

