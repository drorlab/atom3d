---
layout: default
---

## Mutation Stability Prediction (MSP)
  - **Impact:** Identifying mutations that stabilize a protein’s interactions is a key task in designing new proteins. Experimental techniques for probing these are labor intensive, motivating the development of efficient computational methods.
  - **Dataset description:** We derive a novel dataset by collecting single-point mutations from the SKEMPI database (Jankauskaitė et al., 2019) and model each mutation into the structure to produce mutated structures.
  - **Task:** We formulate this as a binary classification task where we predict whether the stability of the complex increases as a result of the mutation.
  - **Splitting criteria:** We split protein complexes by sequence identity at 30%.
  - **Downloads:**

    - Full dataset [[dataset]]('https://drive.google.com/uc?export=download&id=1ACkgojNUKo_ck34F3VEvsjHtlqIs2ecx')
    - 30% sequence identity split
      [[datasets]]('https://drive.google.com/uc?export=download&id=1f2GUGRIxR82l5eb8r8OFX7QkST4zbuZ3')
      [[indices]]('https://drive.google.com/uc?export=download&id=1ppUxfz9eMmEzFfyirUxsPeCmZ3OfZ2X6')

### References

Jankauskaite, J., Jiménez-García, B., et al. (2019). Bioinformatics, 35(3), 462–469.
