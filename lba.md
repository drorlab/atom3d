---
layout: default
---

## Ligand Binding Affinity (LBA) 
  - **Impact:** Most therapeutic drugs and many molecules critical for biological signaling take the form of small molecules. Predicting the strength of the protein-small molecule interaction is a challenging but crucial task for drug discovery applications.
  - **Dataset description:** We use the PDBBind database (Wang et al., 2004; Liu et al., 2015), a curated database containing protein-ligand complexes from the PDB and their corresponding binding strengths.
  - **Task:** We predict pK = -log(K), where K is the binding affinity in Molar units.
  - **Splitting criteria:** We split protein-ligand complexes by protein sequence identity at 30% (most stringent) and 60% (less stringent).
  - **Downloads:**

    - Full dataset [[dataset]](https://drive.google.com/uc?export=download&id=1tudH6z5_-LVTIS7k44QcdXO9nPhIgYeS)
    - 30% sequence identity split
      [[datasets]](https://drive.google.com/uc?export=download&id=1P80r0Snq8EcTK36OyBQcZD5KYCaYuIBm)
      [[indices]](https://drive.google.com/uc?export=download&id=1S8xQH0nmOrKvv7vpx3ETr7fzpmSWoN7_)
    - 60% sequence identity split
      [[datasets]](https://drive.google.com/uc?export=download&id=1IM_Fn5dvvqogwGaq0TD0BRWbv3iST-mN)
      [[indices]](https://drive.google.com/uc?export=download&id=1k_3DTRd9GfDnBS0C2yGRTgrS1WeHy-J_)



### References

Wang, R., Fang, X., Lu, Y., & Wang, S. (2004). Journal of Medicinal Chemistry, 47(12), 2977–2980.

Liu, Z., Li, Y., et al. (2015).  Bioinformatics, 31(3), 405–412.

### License

This dataset is licensed under a Creative Commons NonCommercial-NoDerivs (CC-BY-NC-ND) [license](./LICENSE_ND).
