---
layout: default
---

## Protein Structure Ranking (PSR)
  - **Impact:** Proteins are one of the primary workhorses of the cell, and knowing their structure is often critical to understanding (and engineering) their function.
  - **Dataset description:** The Critical Assessment of Structure Prediction (CASP) (Kryshtafovych et al., 2019) is a blind international competition for predicting protein structure.
  - **Task:** We formulate this as a regression task, where we predict the global distance test (GDT_TS) from the true structure for each of the predicted structures submitted in the last 18 years of CASP.
  - **Splitting criteria:** We split structures temporally by competition year.
  - **Downloads:**

    - Full dataset [[dataset]]('https://drive.google.com/uc?export=download&id=1ahFkfqijbLSO9kelRrp6i8TcufXuSoVa')
    - Split by year
      [[datasets]]('https://drive.google.com/uc?export=download&id=1nmiqJLRZMTnbADzkcEUjfO7H9WwT13Ns')
      [[indices]]('https://drive.google.com/uc?export=download&id=12DCP4OF8DrvaP33Yujlydx-E_Xt3POTx')

### References

Kryshtafovych, A., Schwede, T., et al. (2019). Proteins: Structure, Function and Bioinformatics, 87(12), 1011-1020.
