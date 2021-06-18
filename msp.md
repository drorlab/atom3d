---
layout: default
---

## Mutation Stability Prediction (MSP)
  - **Impact:** Identifying mutations that stabilize a protein’s interactions is a key task in designing new proteins. Experimental techniques for probing these are labor intensive, motivating the development of efficient computational methods.
  - **Dataset description:** We derive a novel dataset by collecting single-point mutations from the SKEMPI database (Jankauskaitė et al., 2019) and model each mutation into the structure to produce mutated structures.
  - **Task:** We formulate this as a binary classification task where we predict whether the stability of the complex increases as a result of the mutation.
  - **Splitting criteria:** We split protein complexes by sequence identity at 30%.
  - **Downloads:** The full dataset, split data, and split indices are available for download via [Zenodo](https://zenodo.org/record/4962515) (doi:10.5281/zenodo.4962515)

### References

Jankauskaite, J., Jiménez-García, B., et al. (2019). Bioinformatics, 35(3), 462–469.

### License

This dataset is licensed under a Creative Commons CC-BY [license](./LICENSE).
