---
layout: default
---

## Ligand Binding Affinity (LBA) 
  - **Impact:** Most therapeutic drugs and many molecules critical for biological signaling take the form of small molecules. Predicting the strength of the protein-small molecule interaction is a challenging but crucial task for drug discovery applications.
  - **Dataset description:** We use the PDBBind database (Wang et al., 2004; Liu et al., 2015), a curated database containing protein-ligand complexes from the PDB and their corresponding binding strengths.
  - **Task:** We predict pK = -log(K), where K is the binding affinity in Molar units.
  - **Splitting criteria:** We split protein-ligand complexes by protein sequence identity at 30% (most stringent) and 60% (less stringent).
  - **Downloads:** The full dataset, split data for 30% and 60% sequence identity, and split indices are available for download via [Zenodo](https://zenodo.org/record/4914718) (doi:10.5281/zenodo.4914718)



### References

Wang, R., Fang, X., Lu, Y., & Wang, S. (2004). Journal of Medicinal Chemistry, 47(12), 2977–2980.

Liu, Z., Li, Y., et al. (2015).  Bioinformatics, 31(3), 405–412.

### License

This dataset is licensed under a Creative Commons NonCommercial-NoDerivs (CC-BY-NC-ND) [license](./LICENSE_ND).
