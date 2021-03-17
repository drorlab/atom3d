---
layout: default
---

## Residue Identity (RES)
  - **Impact:** Understanding the structural role of individual amino acids is important for engineering new proteins. We can understand this role by predicting the substitutabilities of different amino acids at a given protein site based on the surrounding structural environment.
  - **Dataset description:** We generate a novel dataset consisting of atomic environments extracted from nonredundant structures in the PDB.
  - **Task:** We formulate this as a classification task where we predict the identity of the amino acid in the center of the environment based on all other atoms.
  - **Splitting criteria:** We split residue environments by domain-level CATH protein topology class.
  - **Downloads:**

    - Full dataset [[dataset]](https://drive.google.com/uc?export=download&id=1nmSNqAyOKof9-76l4gYQvODsEHNZLxv7)
    - Split by CATH 4.2 class
      [[datasets]](https://drive.google.com/uc?export=download&id=1rJEAyyoFN0Y6pgnLJyG0Fy5FKqAopOqC)
      [[indices]](https://drive.google.com/uc?export=download&id=1xOX7HNuDvJib3-wHxh0JFq0LMLLYQu-6)

