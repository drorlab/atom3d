---
layout: default
---

## Protein Structure Ranking (PSR) [[download]](https://drive.google.com/uc?export=download&id=1-Hn2f60BC4aJYGKLCeL_gebXVQYF6ZGS)
  - **Impact:** Proteins are one of the primary workhorses of the cell, and knowing their structure is often critical to understanding (and engineering) their function.
  - **Dataset description:** The Critical Assessment of Structure Prediction (CASP) (Kryshtafovych et al., 2019) is a blind international competition for predicting protein structure.
  - **Task:** We formulate this as a regression task, where we predict the global distance test (GDT_TS) from the true structure for each of the predicted structures submitted in the last 18 years of CASP.
  - **Splitting criteria:** We split structures temporally by competition year.

### References

Kryshtafovych, A., Schwede, T., et al. (2019). Proteins: Structure, Function and Bioinformatics, 87(12), 1011-1020.
