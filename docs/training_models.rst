Machine learning with ATOM3D
============================

ATOM3D makes it easy to train any machine learning model on 3D biomolecular structure data. All ATOM3D datasets are built on top of PyTorch Datasets, making it simple to create dataloaders that work with almost any model architecture out of the box. 
We provide dataloaders for all pre-curated datasets, as well as some base model architectures for three major classes of deep learning methods for 3D molecular learning: graph neural networks (GNNs), 3D convolutional neural networks (3DCNNs), and equivariant neural networks (ENNs).
Please see our `paper <https://arxiv.org/abs/2012.04035>`_ for more details on the specific choice of architecture for each task.

Preparing data for training
****************************

The structures in ATOM3D Datasets are represented in dataframe format (see :doc:`data_formats`), which need to be processed into numeric tensors before they are ready for training.
One way of doing this is to define a dataloader that yields dataframes, which are then converted into tensors (e.g. graphs or voxelized cubes) after they are loaded.
However, this makes automatic batching somewhat complicated, requiring custom functions for collating dataframes into minibatches.

To avoid this, ATOM3D enables the conversion to tensors to happen as the data retrieved from the dataset (i.e. the items returned by a Dataset's ``getitem`` method contain tensors, rather than dataframes). 
Dataloading and minibatching is then simple to do using Pytorch's ``DataLoader`` class (or the Pytorch-Geometric equivalent for graphs).

The conversion itself happens through the *transform* function, which is passed to the Dataset class on instantiation. 
Transform functions for converting dataframes to graphs (e.g. for GNNs) or voxelized cubes (e.g. for 3D CNNs) are provided in :mod:`atom3d.util.transforms`, but any arbitrary transformation function can be defined similarly.

Base models
***************

For general use, we provide base versions of each model type. These models may be useful as proof-of-concept testing on a new dataset, to provide a strong baseline for benchmarking a specially engineered model architecture, or as a template for the design of new model architectures. 
The base models provided are the following:

  * **GCN** (:class:`atom3d.models.gnn.GCN`)
    
    A simple GNN consisting of five layers of graph convolutions as defined by `Kipf and Welling (2017) <https://arxiv.org/pdf/1609.02907.pdf>`_. Each GCN layer is followed by batch normalization and a ReLU nonlinearity. 
    These layers will learn an embedding vector for each node in the network, but it is often necessary to reduce this to a single vector for classification or regression. We provide two ways to do this: (1) global mean pooling over all nodes (default), or (2) extract the embedding of a single node in the graph supplied by the ``select_idx`` argument. 
    
    This network and all other GNNs are implemented using the pytorch-geometric library. This package must be `installed <https://pytorch-geometric.readthedocs.io/en/latest/notes/installation.html>`_ separately, and data passed into the model should be in the `format <https://pytorch-geometric.readthedocs.io/en/latest/modules/data.html#torch_geometric.data.Data>`_ used by pytorch-geometric.
    For converting Dataset items to graphs, see :mod:`atom3d.util.graph`.

    
  * **CNN3D** (:class:`atom3d.models.cnn.CNN3D`)

    A simple convolutional network consisting of six layers of 3D convolutions, each followed by batch normalization, ReLU activation, and optionally dropout. This network uses strided convolutions for downsampling.
    The desired input and output dimensionality must be specified when instantiating the model. 

    The input data is expected to be a voxelized cube with several feature channels, represented as a tensor with 5 dimensions: (batch_size, in_dimension, box_size, box_size, box_size). For converting Dataset items to voxelized tensors, see :mod:`atom3d.util.cnn`.

     
  * **ENN** (:class:`atom3d.models.enn.ENN`)

    This network and all ENNs based on it are implemented using an adapted version of the `Cormorant <https://papers.nips.cc/paper/2019/file/03573b32b2746e6e8ca98b9123f2249b-Paper.pdf>`_ package. Make sure to install it from the `Dror Lab fork <https://github.com/drorlab/cormorant>`_. 
    
    
  * **FeedForward** (:class:`atom3d.models.ff.FeedForward`)

    A basic feed-forward neural network (multi-layer perceptron), with tunable number of hidden layers and layer dimensions. 
    In many cases, the 3D learning methods above are most useful as feature extractors to transform a molecular structure to a single vector, or embedding. 
    For classification and regression tasks, it is then necessary to transform this vector into the desired output dimensionality.
    Although any method could be used instead, this feed-forward network is simple but flexible, and easy to plug into any machine learning pipeline.


Model-specific installation instructions
*****************************************

The standard installation described in the `introduction <https://atom3d.readthedocs.io/en/latest/getting_started.html#installation>`_ lets you use all the data loading and processing functions included in ATOM3D. 
To use the specific machine learning models, additional dependencies can be necessary. Here we describe which ones you need for each type of model.

3DCNN
"""""""""""""""""""""

No additional dependencies are needed to use 3D convolutional neural networks.

GNN
"""""""""""""""""""""

The graph neural networks in ATOM3D are based on `PyTorch Geometric <https://github.com/rusty1s/pytorch_geometric>`_

In order to use it, you should install ATOM3D in a dedicated environment, defining the correct version of the CUDA toolkit (here: 10.2)::

    conda create --name geometric -c pytorch -c rdkit pip rdkit pytorch=1.5 cudatoolkit=10.2
    conda activate geometric
    pip install atom3d

Then install PyTorch Geometric by running::

    pip install torch-scatter==latest+${CUDA} -f https://pytorch-geometric.com/whl/torch-1.5.0.html
    pip install torch-sparse==latest+${CUDA} -f https://pytorch-geometric.com/whl/torch-1.5.0.html
    pip install torch-cluster==latest+${CUDA} -f https://pytorch-geometric.com/whl/torch-1.5.0.html
    pip install torch-spline-conv==latest+${CUDA} -f https://pytorch-geometric.com/whl/torch-1.5.0.html
    pip install torch-geometric

where `${CUDA}` should be replaced by either `cpu`, `cu92`, `cu101` or `cu102` depending on your PyTorch installation (`cu102` if installed as above).

If you do not know your CUDA version, you can find out via::

    nvcc --version
    

ENN
"""""""""""""""""""""

The equivariant networks in ATOM3D are based on `Cormorant <https://arxiv.org/abs/1906.04015>`_.

In order to use it, you should install ATOM3D in a dedicated environment, defining the correct version of the CUDA toolkit (here: 10.1)::

    conda create --name cormorant python=3.7 pip scipy pytorch cudatoolkit=10.1 -c pytorch
    conda activate cormorant
    pip install atom3d

If you do not know your CUDA version, you can find out via::

    nvcc --version

The Cormorant fork used for this project can be cloned directly from the git repo using::

    cd ~
    git clone https://github.com/drorlab/cormorant.git


You can currently only install it in development mode by going to the directory with setup.py and running::

    cd cormorant
    python setup.py develop


Task-specific models and dataloaders
*************************************

Many datasets and tasks require specific model architectures (e.g. paired or multi-headed networks), and thus require custom-built dataloaders to process the data in the correct manner. 
We provide custom dataloaders and models for each pre-curated dataset in the :mod:`atom3d.datasets` module. A brief description of each is provided below; for more details and motivation please see the ATOM3D `paper <https://arxiv.org/abs/2012.04035>`_.

  * **SMP** (:mod:`atom3d.datasets.smp.models`)

  * **PIP** (:mod:`atom3d.datasets.pip.models`)

  * **RES** (:mod:`atom3d.datasets.res.models`)

  * **MSP** (:mod:`atom3d.datasets.msp.models`)

  * **LBA** (:mod:`atom3d.datasets.lba.models`)

  * **LEP** (:mod:`atom3d.datasets.lep.models`)

  * **PSR** (:mod:`atom3d.datasets.psr.models`)

  * **RSR** (:mod:`atom3d.datasets.rsr.models`)

Examples
**********

1. **Train base GCN model on a protein dataset, with default parameters.**

In this example, the dataset contains labels for each example under the ``label`` key. 
These are assumed to be binary labels applied to the entire graph, rather than to a specific node.

The underlying dataset contains dataframes in the ``atoms`` field, as with all ATOM3D Datasets, but for training we must convert these to tensors representing each graph. 
This is done via the `transform` function, which enables automatic batching via ``DataLoader`` objects (either standard Pytorch or Pytorch-Geometric).

We are assuming a binary classification problem, and using the GCN as a feature extractor. 
Therefore, we need a model to transform from the feature representation to the output prediction (a single value).
This example uses a simple feed-forward neural network with one hidden layer.

  .. code:: python

    # pytorch imports
    import torch
    import torch.nn as nn
    from torch_geometric.data import Data, DataLoader
    
    # atom3d imports
    import atom3d.datasets.datasets as da
    import atom3d.util.graph as gr
    import atom3d.util.transforms as tr
    from atom3d.models.gnn import GCN
    from atom3d.models.mlp import MLP
    
    # define training hyperparameters
    learning_rate=1e-4
    epochs = 5
    feat_dim = 128
    out_dim = 1
    
    # Load dataset (with transform to convert dataframes to graphs) and initialize dataloader
    dataset = da.load_dataset('data/test_lmdb', 'lmdb', transform=tr.GraphTransform(atom_key='atoms', label_key='label'))
    dataloader = DataLoader(dataset, batch_size=2, shuffle=True)
    
    # get number of input features from first graph
    for batch in dataloader:
        in_dim = batch.num_features
        break
    
    # GCN feature extraction module
    feat_model = GCN(in_dim, feat_dim)
    # Feed-forward output module
    out_model = MLP(feat_dim, [64], out_dim)
    
    # define optimizer and criterion
    params = [x for x in feat_model.parameters()] + [x for x in out_model.parameters()]
    optimizer = torch.optim.Adam(params, lr=1e-4)
    criterion = nn.BCEWithLogitsLoss()
    
    # Training loop
    for epoch in range(epochs):
        for batch in dataloader:
            # labels need to be float for BCE loss
            labels = batch.y.float()
            # calculate 128-dim features
            feats = feat_model(batch.x, batch.edge_index, batch.edge_attr, batch.batch)
            # calculate predictions
            out = out_model(feats)
            # compute loss and backprop
            loss = criterion(out.view(-1), labels)
            loss.backward()
            optimizer.step()
        print('Epoch {}: train loss {}'.format(epoch, loss))

2. **Train an equivariant model on ligand binding affinity.**

We provide several example training scripts (+ model implementations and dataloaders) in a dedicated folder in the GitHub repository.
One particularly educative example is training of an  `equivariant neural network on ligand binding affinity data <https://github.com/drorlab/atom3d/tree/master/examples/lba/enn>`_.
