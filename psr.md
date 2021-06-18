---
layout: default
---

## Protein Structure Ranking (PSR)
  - **Impact:** Proteins are one of the primary workhorses of the cell, and knowing their structure is often critical to understanding (and engineering) their function.
  - **Dataset description:** The Critical Assessment of Structure Prediction (CASP) (Kryshtafovych et al., 2019) is a blind international competition for predicting protein structure.
  - **Task:** We formulate this as a regression task, where we predict the global distance test (GDT_TS) from the true structure for each of the predicted structures submitted in the last 18 years of CASP.
  - **Splitting criteria:** We split structures temporally by competition year.
  - **Downloads:** The full dataset, split data, and split indices are available for download via [Zenodo](https://zenodo.org/record/4915648) (doi:10.5281/zenodo.4915648)

### References

Kryshtafovych, A., Schwede, T., et al. (2019). Proteins: Structure, Function and Bioinformatics, 87(12), 1011-1020.

### License

This dataset is licensed under a Creative Commons CC-BY [license](./LICENSE).
