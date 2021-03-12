Quick Start
===========

Introduction
************

ATOM3D aims to facilitate the development of novel machine learning methods on three-dimensional molecular structure by providing several curated benchmark datasets across a variety of tasks in molecular and strucural biology. This package provides a set of standardized functions and methodologies for interacting with provided datasets as well as preparing new 3D molecular datasets.

Installation
******************

Install using pip
"""""""""""""""""""

  .. code:: bash
    
    pip install atom3d

Install from source
"""""""""""""""""""""

To install, first clone the ATOM3D repository:

  .. code:: bash

     git clone https://github.com/drorlab/atom3d

To install with base dependencies:

   .. code:: bash

      make requirements

To install with RDKit (needed for processing small molecule files, e.g. SDF/MOL2), install within conda:

   .. code:: bash

      conda create -n atom3d python=3.6 pip rdkit
      conda activate atom3d
      make requirements
      
      
Installation for Specific Models
********************************

The standard installation described above lets you use all the data loading and processing functions included in ATOM3D. 
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



Frequently Asked Questions
**************************

.. _sec:datasets-faq:

1. **What pre-curated datasets are available through ATOM3D?**
""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""

   | ATOM3D currently contains eight datasets, spanning molecular structure, function, interaction, and design tasks:

     * *Small Molecule Properties (SMP)*

       Predicting physico-chemical properties of small molecules is a common task in medicinal chemistry and materials design. SMP is based on the QM9 dataset, which contains structures and energetic, electronic, and thermodynamic properties for 134,000 stable small organic molecules, obtained from quantum-chemical calculations.


     * *Protein Interface Prediction (PIP)*
     
       Proteins interact with each other in many scenarios—for example, our antibody proteins recognize diseases by binding to antigens. A critical problem in understanding these interactions is to identify which amino acids of two given proteins will interact upon binding. The PIP dataset contains structures from the Database of Interacting Protein Structures (DIPS), a comprehensive dataset of protein complexes mined from the PDB, and the Docking Benchmark 5 (DB5), a smaller gold standard dataset.
     
     
     * *Residue Identity (RES)*
     
       Understanding the structural role of individual amino acids is important for engineering new proteins. We can understand this role by predicting the substitutabilities of different amino acids at a given protein site based on the surrounding structural environment. The RES dataset consists of atomic environments extracted from nonredundant structures in the PDB.
     
     
     * *Mutation Stability Prediction (MSP)*
     
       Identifying mutations that stabilize a protein’s interactions is a key task in designing new proteins. Experimental techniques for probing these are labor intensive, motivating the development of efficient computational methods. MSP contains structures from the SKEMPI dataset of protein-protein interactions, with each mutation computationally modeled into the structure.
     
     
     * *Ligand Binding Affinity (LBA)*
     
       Most therapeutic drugs and many molecules critical for biological signaling take the form of small molecules. Predicting the strength of the protein-small molecule interaction is a challenging but crucial task for drug discovery applications. LBA contains structures from the "refined set" of PDBBind, a curated database containing protein-ligand complexes from the PDB and their corresponding binding strengths.
     
     
     * *Ligand Efficacy Prediction (LEP)*
     
       Many proteins switch on or off their function by changing shape. Predicting which shape a drug will favor is thus an important task in drug design. LEP contains a curated set of proteins from several families with both ”active” and ”inactive” state structures, with 527 small molecules with known activating or inactivating function modeled in using the program Glide.
     
     
     * *Protein Structure Ranking (PSR)*
     
       Assessing the quality of a specific 3D protein conformation is a crucial aspect of computational protein structure prediction. PSR contains data from the Critical Assessment of Structure Prediction (CASP), a blind international competition for predicting protein structure.
     
     
     * *RNA Structure Ranking (RSR)*
     
       Similar to proteins, RNA plays major functional roles (e.g., gene regulation) and can adopt well-defined 3D shapes. However the problem is data-poor, with only a few hundred known structures. PSR contains candidate structures for the first 21 released RNA Puzzle challenges, a blind structure prediction competition for RNA.
     
.. _sec:download-faq:

2. **How do I download the ATOM3D datasets?**
"""""""""""""""""""""""""""""""""""""""""""""""
   
   | All datasets can be downloaded in LMDB format from `atom3d.ai <atom3d.ai>`_, or using the Python API:
     
     .. code:: pycon
   
        >>> import atom3d.datasets.datasets as da
        >>> da.download_dataset('lba', '/path/to/target')

3. **Do I have to use the provided train/val/test splits for ATOM3D datasets?**
""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""

   | No, you may create your own splitting functions and apply them to any dataset. Please see :doc:`/using_datasets` for more details.


4. **What kind of utility functions exist in ATOM3D?**
""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""

   | There are functions available for performing many common tasks on macromolecular structure. See the :ref:`usage examples <examples>` for some common use cases, and explore the API documentation to find specific functions. 

   | If we are missing a function you think would be useful, please consider :doc:`contributing </contributing>`!

5. **Can I contribute datasets and models back to ATOM3D?**
""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""

   | Yes!  We are happy to accept new datasets and models!  See :doc:`contributing </contributing>` for details.

Reference
*************
If you use ATOM3D in your work, please cite our preprint:

Townshend, R. J. L., Vögele, M., Suriana, P., Derry, A., Powers, A., Laloudakis, Y., Balachandar, S., Anderson, B., Eismann, S., Kondor, R., Altman, R. B., Dror, R. O. (2020). ATOM3D: Tasks On Molecules in Three Dimensions. *arXiv:2012.04035*. http://arxiv.org/abs/2012.04035.

For specific datasets, please also cite the respective original source(s) as specified in the preprint.
