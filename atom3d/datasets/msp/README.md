# Mutation Stability Prediction

## Overview

The Mutation Stability Prediction (MSP) task involves classifying whether
mutations in the SKEMPI 2.0 database (J. Jankauskaite, B. Jiménez-García et al., 2019) are stabilizing or not using the provided protein structures.

Each mutation in the MSP task includes a PDB file with the residue of interest transformed to the specified mutant amino acid as well as the native PDB file.

A total of 4148 mutant structures accompanied by their 316 WT structures are provided.

## Notes

Non-point mutations or mutants that caused non-binding of the complex were screened out from SKEMPI.
Additionally, mutations involving a disulfide bond and mutants from the PDBs 1KBH or 1JCK were ignored due to processing difficulties. A label of 1 was assigned to a mutant if the Kd of the mutant protein was less than that of the wild-type protein, indicating better binding, and 0 otherwise. If a mutant had multiple conflicting stability results in the original data, the first result seen when processing the SKEMPI dataset was used. 

## Datasets

TBD

### Format

## Usage

TBD
