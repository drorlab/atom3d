## ATOM3D: Tasks on Molecules in Three Dimensions

ATOM3D is a unified collection of datasets concerning the three-dimensional structure of biomolecules, including proteins, small molecules, and nucleic acids. These datasets are specifically designed to provide a benchmark for machine learning methods which operate on 3D molecular structure, and represent a variety of important structural, functional, and engineering tasks. All datasets are provided in a standardized format along with corresponding processing code and dataloaders for common machine learning frameworks (PyTorch and TensorFlow). ATOM3D is designed to be a living database, where datasets are updated and tasks are added as the field progresses.

- **Repository**: All dataset processing code and installation instructions can be found at [https://github.com/drorlab/atom3d](https://github.com/drorlab/atom3d)

- **Paper**: Please see our [preprint](arxiv.org/XXXX) for further details on the datasets and benchmarks.

- **NeurIPS LMRL Workshop 2020**: Check out our poster and talk at the Learning Meaningful Representations of Life workshop at NeurIPS 2020.

### Datasets
  
ATOM3D currently contains eight datasets, which can be roughly grouped into four categories that represent a wide range of problems, spanning single molecular structures and interactions between biomolecules as well as molecular functional and design/engineering tasks. Click on the corresponding dataset to download as `tar.gz` file.

<svg version="1.1" xmlns="http://www.w3.org/2000/svg" xmlns:xlink="http://www.w3.org/1999/xlink" viewBox="0 0 3600 2400">
  <image width="3600" height="2400" xlink:href="composite_Datasets.png"></image>
	<a xlink:href="https://drive.google.com/uc?export=download&id=1Uce6a6VoN9gYAn3V4eR3QC0f6a6mOXpI" alt="SMP">
		<rect x="1200" y="1675" fill="#fff" opacity="0" width="440" height="660" />
	</a>
	<a xlink:href="https://drive.google.com/uc?export=download&id=1EL4ybt2SJF7iLzbavBGlU1ImMiZ0dOkH" alt="PIP">
		<rect x="2650" y="360" fill="#fff" opacity="0" width="885" height="780" />
	</a>
	<a xlink:href="https://drive.google.com/uc?export=download&id=1CzLiTDFgApIBaI1znLjEk2d3T0Zh4Yjo" alt="RES">
		<rect x="1890" y="1675" fill="#fff" opacity="0" width="730" height="700" />
	</a>
	<a xlink:href="https://drive.google.com/uc?export=download&id=186MLykFkC3IbslXhLfHIwQwnDOy1Sr49" alt="MSP">
		<rect x="2660" y="1265" fill="#fff" opacity="0" width="880" height="630" />
	</a>
  	<a xlink:href="https://drive.google.com/uc?export=download&id=1pj0RCW3mOMnB2FYQPmMv6XFMS0Ps7RvY" alt="LBA">
		<rect x="1910" y="45" fill="#fff" opacity="0" width="740" height="610" />
	</a>
	<a xlink:href="https://drive.google.com/uc?export=download&id=1NykcNi0im_XfUK4NuO-g4LlsSJoQl7jQ" alt="LEP">
		<rect x="70" y="1270" fill="#fff" opacity="0" width="935" height="750" />
	</a>
	<a xlink:href="https://drive.google.com/uc?export=download&id=1-Hn2f60BC4aJYGKLCeL_gebXVQYF6ZGS" alt="PSR">
		<rect x="100" y="350" fill="#fff" opacity="0" width="895" height="830" />
	</a>
	<a xlink:href="https://drive.google.com/uc?export=download&id=1imQiQI6kyDnA4t-rxju0PetgJsASkx7S" alt="RSR">
		<rect x="820" y="45" fill="#fff" opacity="0" width="930" height="660" />
	</a>
	
	
</svg>


#### Small Molecule Properties (SMP) [[download]](https://drive.google.com/uc?export=download&id=1Uce6a6VoN9gYAn3V4eR3QC0f6a6mOXpI)
  - **Impact:** Predicting physico-chemical properties of small molecules is a common task in medicinal chemistry and materials design. Quantum chemical calculations can save expensive experiments but are themselves costly and cannot cover the huge chemical space spanned by candidate molecules. 
  - **Dataset description:** We use the QM9 dataset (Ruddigkeit et al., 2012; Ramakrishnan et al., 2014), which contains structures and energetic, electronic, and thermodynamic properties for 134,000 stable small organic molecules, obtained from quantum-chemical calculations. 
  - **Task:** We predict all molecular properties from the ground-state structure.
  - **Splitting criteria:** We split molecules randomly.
  
#### Protein Interface Prediction (PIP) [[download]](https://drive.google.com/uc?export=download&id=1EL4ybt2SJF7iLzbavBGlU1ImMiZ0dOkH)
  - **Impact:** Proteins interact with each other in many scenarios—for example, our antibody proteins recognize diseases by binding to antigens. A critical problem in understanding these interactions is to identify which amino acids of two given proteins will interact upon binding.
  - **Dataset description:** For training, we use the Database of Interacting Protein Structures (DIPS), a comprehensive dataset of protein complexes mined from the PDB (Townshend et al., 2019). We predict on the Docking Benchmark 5 (Vreven et al., 2015), a smaller gold standard dataset.
  - **Task:** We predict if two amino acids will come into contact when their respective proteins bind.
  - **Splitting criteria:** We split protein complexes by sequence identity at 30%.

#### Residue Identity (RES) [[download]](https://drive.google.com/uc?export=download&id=1CzLiTDFgApIBaI1znLjEk2d3T0Zh4Yjo)
  - **Impact:** Understanding the structural role of individual amino acids is important for engineering new proteins. We can understand this role by predicting the substitutabilities of different amino acids at a given protein site based on the surrounding structural environment.
  - **Dataset description:** We generate a novel dataset consisting of atomic environments extracted from nonredundant structures in the PDB.
  - **Task:** We formulate this as a classification task where we predict the identity of the amino acid in the center of the environment based on all other atoms.
  - **Splitting criteria:** We split residue environments by domain-level CATH protein topology class.

#### Mutation Stability Prediction (MSP) [[download]](https://drive.google.com/uc?export=download&id=186MLykFkC3IbslXhLfHIwQwnDOy1Sr49)
  - **Impact:** Identifying mutations that stabilize a protein’s interactions is a key task in designing new proteins. Experimental techniques for probing these are labor intensive, motivating the development of efficient computational methods.
  - **Dataset description:** We derive a novel dataset by collecting single-point mutations from the SKEMPI database (Jankauskaitė et al., 2019) and model each mutation into the structure to produce mutated structures.
  - **Task:** We formulate this as a binary classification task where we predict whether the stability of the complex increases as a result of the mutation.
  - **Splitting criteria:** We split protein complexes by sequence identity at 30%.

#### Ligand Binding Affinity (LBA) [[download]](https://drive.google.com/uc?export=download&id=1pj0RCW3mOMnB2FYQPmMv6XFMS0Ps7RvY)
  - **Impact:** Most therapeutic drugs and many molecules critical for biological signaling take the form of small molecules. Predicting the strength of the protein-small molecule interaction is a challenging but crucial task for drug discovery applications.
  - **Dataset description:** We use the PDBBind database (Wang et al., 2004; Liu et al., 2015), a curated database containing protein-ligand complexes from the PDB and their corresponding binding strengths.
  - **Task:** We predict pK = -log(K), where K is the binding affinity in Molar units.
  - **Splitting criteria:** We split protein-ligand complexes by protein sequence identity at 30%.

#### Ligand Efficacy Prediction (LEP) [[download]](https://drive.google.com/uc?export=download&id=1NykcNi0im_XfUK4NuO-g4LlsSJoQl7jQ)
  - **Impact:** Many proteins switch on or off their function by changing shape. Predicting which shape a drug will favor is thus an important task in drug design.
  - **Dataset description:** We develop a novel dataset by curating proteins from several families with both ”active” and ”inactive” state structures, and model in 527 small molecules with known activating or inactivating function using the program Glide (Friesner et al., 2004).
  - **Task:** We formulate this as a binary classification task where we predict whether or not a molecule bound to the structures will be an activator of the protein’s function or not.
  - **Splitting criteria:** We split complex pairs by protein.
  
#### Protein Structure Ranking (PSR) [[download]](https://drive.google.com/uc?export=download&id=1-Hn2f60BC4aJYGKLCeL_gebXVQYF6ZGS)
  - **Impact:** Proteins are one of the primary workhorses of the cell, and knowing their structure is often critical to understanding (and engineering) their function.
  - **Dataset description:** The Critical Assessment of Structure Prediction (CASP) (Kryshtafovych et al., 2019) is a blind international competition for predicting protein structure.
  - **Task:** We formulate this as a regression task, where we predict the global distance test (GDT_TS) from the true structure for each of the predicted structures submitted in the last 18 years of CASP.
  - **Splitting criteria:** We split structures temporally by competition year.

#### RNA Structure Ranking (RSR) [[download]](https://drive.google.com/uc?export=download&id=1imQiQI6kyDnA4t-rxju0PetgJsASkx7S)
  - **Impact:** Similar to proteins, RNA plays major functional roles (e.g., gene regulation) and can adopt well-defined 3D shapes. Yet the problem is data-poor, with only a few hundred known structures.
  - **Dataset description:** Candidate models generated by FARFAR2 (Watkins & Das, 2019) for the first 21 released RNA Puzzle challenges (Cruz et al., 2012), a blind structure prediction competition for RNA.
  - **Task:** We predict the root-mean-squared deviation (RMSD) from the ground truth structure.
  - **Splitting criteria:** We split structures temporally by competition year.

### References

Cruz, J. A., Blanchet, M. F., et al. (2012). RNA, 18(4), 610–625.

Friesner, R. A., Banks, J. L., et al. (2004). Journal of Medicinal Chemistry, 47(7), 1739–1749.

Jankauskaite, J., Jiménez-García, B., et al. (2019). Bioinformatics, 35(3), 462–469.

Kryshtafovych, A., Schwede, T., et al. (2019). Proteins: Structure, Function and Bioinformatics, 87(12), 1011-1020.

Liu, Z., Li, Y., et al. (2015).  Bioinformatics, 31(3), 405–412.

Ramakrishnan, R., Dral, P. O., et al. (2014). Scientific Data, 1:1-7.

Ruddigkeit, L., Van Deursen, R., et al. (2012). Journal of Chemical Information and Modeling, 52(11), 2864–2875. 

Townshend, R. J. L., Bedi, R., et al. (2019).  Advances in Neural Information Processing Systems (Vol. 32, pp. 15616–15625).

Vreven, T., Moal, I. H., et al. (2015). Journal of Molecular Biology, 427(19), 3031–3041.

Wang, R., Fang, X., Lu, Y., & Wang, S. (2004). Journal of Medicinal Chemistry, 47(12), 2977–2980.

Watkins, A. M., Rangan, R., & Das, R. (2020). Structure, 28(8), 963-976.e6. 
