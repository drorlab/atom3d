---
layout: default
---

## Ligand Binding Affinity (LBA) [[download]](https://drive.google.com/uc?export=download&id=1pj0RCW3mOMnB2FYQPmMv6XFMS0Ps7RvY)
  - **Impact:** Most therapeutic drugs and many molecules critical for biological signaling take the form of small molecules. Predicting the strength of the protein-small molecule interaction is a challenging but crucial task for drug discovery applications.
  - **Dataset description:** We use the PDBBind database (Wang et al., 2004; Liu et al., 2015), a curated database containing protein-ligand complexes from the PDB and their corresponding binding strengths.
  - **Task:** We predict pK = -log(K), where K is the binding affinity in Molar units.
  - **Splitting criteria:** We split protein-ligand complexes by protein sequence identity at 30%.

### References

Wang, R., Fang, X., Lu, Y., & Wang, S. (2004). Journal of Medicinal Chemistry, 47(12), 2977–2980.

Liu, Z., Li, Y., et al. (2015).  Bioinformatics, 31(3), 405–412.

