---
layout: default
---

## Protein-Protein Interfaces (PPI)
  - **Impact:** Proteins interact with each other in many scenarios—for example, our antibody proteins recognize diseases by binding to antigens. A critical problem in understanding these interactions is to identify which amino acids of two given proteins will interact upon binding.
  - **Dataset description:** For training, we use the Database of Interacting Protein Structures (DIPS), a comprehensive dataset of protein complexes mined from the PDB (Townshend et al., 2019). We predict on the Docking Benchmark 5 (Vreven et al., 2015), a smaller gold standard dataset.
  - **Task:** We predict if two amino acids will come into contact when their respective proteins bind.
  - **Splitting criteria:** We split protein complexes by sequence identity at 30%.
  - **Downloads:**

    - Full dataset [[dataset]]('https://drive.google.com/uc?export=download&id=1QYAXy71s9oStaSBnaVIL0i62jNSpiGQB')
    - DIPS split at 30% sequence identity
      [[datasets]]('https://drive.google.com/uc?export=download&id=1ddUdYTr5aqXJv0Ncz1TWloqiLCLPLO_K')
      [[indices]]('https://drive.google.com/uc?export=download&id=1X7Y4S_QXRFGo3VyL1OroOHt_4YiE5Sfl')

### References

Townshend, R. J. L., Bedi, R., et al. (2019).  Advances in Neural Information Processing Systems (Vol. 32, pp. 15616–15625).

Vreven, T., Moal, I. H., et al. (2015). Journal of Molecular Biology, 427(19), 3031–3041.
