---
layout: default
---

## Ligand Efficacy Prediction (LEP) [[download]](https://drive.google.com/uc?export=download&id=1NykcNi0im_XfUK4NuO-g4LlsSJoQl7jQ)
  - **Impact:** Many proteins switch on or off their function by changing shape. Predicting which shape a drug will favor is thus an important task in drug design.
  - **Dataset description:** We develop a novel dataset by curating proteins from several families with both ”active” and ”inactive” state structures, and model in 527 small molecules with known activating or inactivating function using the program Glide (Friesner et al., 2004).
  - **Task:** We formulate this as a binary classification task where we predict whether or not a molecule bound to the structures will be an activator of the protein’s function or not.
  - **Splitting criteria:** We split complex pairs by protein.

### References

Friesner, R. A., Banks, J. L., et al. (2004). Journal of Medicinal Chemistry, 47(7), 1739–1749.
