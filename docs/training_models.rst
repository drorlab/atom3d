Machine learning with ATOM3D
============================

ATOM3D makes it easy to train any machine learning model on 3D biomolecular structure data. All ATOM3D datasets are built on top of PyTorch Datasets, making it simple to create dataloaders that work with almost any model architecture out of the box. 
We provide dataloaders for all pre-curated datasets, as well as some base model architectures for three major classes of deep learning methods for 3D molecular learning: graph neural networks (GNNs), 3D convolutional neural networks (3DCNNs), and equivariant neural networks (ENNs).
Please see our `paper <https://arxiv.org/abs/2012.04035>`_ for more details on the specific choice of architecture for each task.


Base models
***************

For general use, we provide base versions of each model type. These models may be useful as proof-of-concept testing on a new dataset, to provide a strong baseline for benchmarking a specially engineered model architecture, or as a template for the design of new model architectures. 
The base models provided are the following:

  * **GCN** (:class:`atom3d.models.gnn.GCN`)
    
    A simple GNN consisting of five layers of graph convolutions as defined by `Kipf and Welling (2017) <https://arxiv.org/pdf/1609.02907.pdf>`_. Each GCN layer is followed by batch normalization and a ReLU nonlinearity. 
    These layers will learn an embedding vector for each node in the network, but it is often necessary to reduce this to a single vector for classification or regression. We provide two ways to do this: (1) global mean pooling over all nodes (default), or (2) extract the embedding of a single node in the graph supplied by the ``select_idx`` argument. 
    
    This network and all other GNNs are implemented using the pytorch-geometric library. This package must be `installed <https://pytorch-geometric.readthedocs.io/en/latest/notes/installation.html>`_ separately, and data passed into the model should be in the `format <https://pytorch-geometric.readthedocs.io/en/latest/modules/data.html#torch_geometric.data.Data>`_ used by pytorch-geometric.

    
  * **CNN3D** (:class:`atom3d.models.cnn.CNN3D`)

    A simple convolutional network consisting of six layers of 3D convolutions, each followed by batch normalization, ReLU activation, and optionally dropout. This network uses strided convolutions for downsampling.
    The desired input and output dimensionality must be specified when instantiating the model.

  * **ENN** (:class:`atom3d.models.enn.ENN`)

    A Cormorant implementation of equivariant networks will be made available soon.


Task-specific models and dataloaders
*************************************

