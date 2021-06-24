---
layout: default
---

## Residue Identity (RES)
  - **Impact:** Understanding the structural role of individual amino acids is important for engineering new proteins. We can understand this role by predicting the substitutabilities of different amino acids at a given protein site based on the surrounding structural environment.
  - **Dataset description:** We generate a novel dataset consisting of atomic environments extracted from nonredundant structures in the PDB.
  - **Task:** We formulate this as a classification task where we predict the identity of the amino acid in the center of the environment based on all other atoms.
  - **Splitting criteria:** We split residue environments by domain-level CATH protein topology class.
  - **Downloads:** The full dataset, split data, and split indices are available for download via [Zenodo](htt    ps://zenodo.org/record/5026743) (doi:10.5281/zenodo.5026743)

### License

This dataset is licensed under a Creative Commons CC-BY [license](./LICENSE).
