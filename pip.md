---
layout: default
---

## Protein Interface Prediction (PIP) [[download]](https://drive.google.com/uc?export=download&id=1EL4ybt2SJF7iLzbavBGlU1ImMiZ0dOkH)
  - **Impact:** Proteins interact with each other in many scenarios—for example, our antibody proteins recognize diseases by binding to antigens. A critical problem in understanding these interactions is to identify which amino acids of two given proteins will interact upon binding.
  - **Dataset description:** For training, we use the Database of Interacting Protein Structures (DIPS), a comprehensive dataset of protein complexes mined from the PDB (Townshend et al., 2019). We predict on the Docking Benchmark 5 (Vreven et al., 2015), a smaller gold standard dataset.
  - **Task:** We predict if two amino acids will come into contact when their respective proteins bind.
  - **Splitting criteria:** We split protein complexes by sequence identity at 30%.

### References

Townshend, R. J. L., Bedi, R., et al. (2019).  Advances in Neural Information Processing Systems (Vol. 32, pp. 15616–15625).

Vreven, T., Moal, I. H., et al. (2015). Journal of Molecular Biology, 427(19), 3031–3041.
