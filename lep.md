---
layout: default
---

## Ligand Efficacy Prediction (LEP)
  - **Impact:** Many proteins switch on or off their function by changing shape. Predicting which shape a drug will favor is thus an important task in drug design.
  - **Dataset description:** We develop a novel dataset by curating proteins from several families with both ”active” and ”inactive” state structures, and model in 527 small molecules with known activating or inactivating function using the program Glide (Friesner et al., 2004).
  - **Task:** We formulate this as a binary classification task where we predict whether or not a molecule bound to the structures will be an activator of the protein’s function or not.
  - **Splitting criteria:** We split complex pairs by protein.
  - **Downloads:**

    - Full dataset [[dataset]](https://drive.google.com/uc?export=download&id=1RrFTAt7ELazQTiMV78Bir136xrydc_Wp)
    - Split by protein
      [[datasets]](https://drive.google.com/uc?export=download&id=1FunndWKkA9sdIP28Qg_LvphGTO1KwJ7w)
      [[indices]](https://drive.google.com/uc?export=download&id=1ZsGTv8t_QwQLYOmHADM0xRb7InFu3M3j)

### References

Friesner, R. A., Banks, J. L., et al. (2004). Journal of Medicinal Chemistry, 47(7), 1739–1749.
