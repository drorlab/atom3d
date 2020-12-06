## ATOM3D: Tasks on Molecules in Three Dimensions

ATOM3D is a unified collection of datasets concerning the three-dimensional structure of biomolecules, including proteins, small molecules, and nucleic acids. These datasets are specifically designed to provide a benchmark for machine learning methods which operate on 3D molecular structure, and represent a variety of important structural, functional, and engineering tasks. All datasets are provided in a standardized format along with corresponding processing code and dataloaders for common machine learning frameworks (PyTorch and TensorFlow). ATOM3D is designed to be a living database, where datasets are updated and tasks are added as the field progresses.

Current datasets include:
  - Small Molecule Properties (SMP) [[download]](https://drive.google.com/uc?export=download&id=1Uce6a6VoN9gYAn3V4eR3QC0f6a6mOXpI)
  - Protein Interface Prediction (PIP) [[download]](https://drive.google.com/uc?export=download&id=1EL4ybt2SJF7iLzbavBGlU1ImMiZ0dOkH)
  - Residue Identity (RES) [[download]](https://drive.google.com/uc?export=download&id=1CzLiTDFgApIBaI1znLjEk2d3T0Zh4Yjo)
  - Mutation Stability Prediction (MSP) [[download]](https://drive.google.com/uc?export=download&id=186MLykFkC3IbslXhLfHIwQwnDOy1Sr49)
  - Ligand Binding Affinity (LBA) [[download]](https://drive.google.com/uc?export=download&id=1pj0RCW3mOMnB2FYQPmMv6XFMS0Ps7RvY)
  - Ligand Efficacy Prediction (LEP) [[download]](https://drive.google.com/uc?export=download&id=1NykcNi0im_XfUK4NuO-g4LlsSJoQl7jQ)
  - Protein Structure Ranking (PSR) [[download]](https://drive.google.com/uc?export=download&id=1-Hn2f60BC4aJYGKLCeL_gebXVQYF6ZGS)
  - RNA Structure Ranking (RSR) [[download]](https://drive.google.com/uc?export=download&id=1imQiQI6kyDnA4t-rxju0PetgJsASkx7S)
  
These datasets can be roughly grouped into four categories that represent a wide range of problems, spanning single molecular structures and interactions between biomolecules as well as molecular functional and design/engineering tasks.

<img src="composite_Datasets.png" usemap="#image-map">

<map name="image-map">
    <area target="_self" alt="SMP" title="SMP" href="https://drive.google.com/uc?export=download&amp;id=1Uce6a6VoN9gYAn3V4eR3QC0f6a6mOXpI" coords="1218,2352,1657,1669" shape="rect">
    <area target="_self" alt="PIP" title="PIP" href="https://drive.google.com/uc?export=download&amp;id=1EL4ybt2SJF7iLzbavBGlU1ImMiZ0dOkH" coords="2668,1138,3597,346" shape="rect">
    <area target="_self" alt="RES" title="RES" href="https://drive.google.com/uc?export=download&amp;id=1CzLiTDFgApIBaI1znLjEk2d3T0Zh4Yjo" coords="1889,1663,2620,2384" shape="rect">
    <area target="_self" alt="MSP" title="MSP" href="https://drive.google.com/uc?export=download&amp;id=186MLykFkC3IbslXhLfHIwQwnDOy1Sr49" coords="2633,1256,3571,1898" shape="rect">
    <area target="_self" alt="LBA" title="LBA" href="https://drive.google.com/uc?export=download&amp;id=1pj0RCW3mOMnB2FYQPmMv6XFMS0Ps7RvY" coords="1899,29,2659,652" shape="rect">
    <area target="_self" alt="LEP" title="LEP" href="https://drive.google.com/uc?export=download&amp;id=1NykcNi0im_XfUK4NuO-g4LlsSJoQl7jQ" coords="80,1240,1024,1999" shape="rect">
    <area target="_self" alt="RSR" title="RSR" href="https://drive.google.com/uc?export=download&amp;id=1imQiQI6kyDnA4t-rxju0PetgJsASkx7S" coords="731,22,843,404,1056,728,1492,709,1784,346,1771,29" shape="poly">
    <area target="_self" alt="PSR" title="PSR" href="https://drive.google.com/uc?export=download&amp;id=1-Hn2f60BC4aJYGKLCeL_gebXVQYF6ZGS" coords="270,305,786,318,1062,798,1024,1167,92,1163" shape="poly">
</map>

### References

- **Repository**: All dataset processing code and installation instructions can be found at [https://github.com/drorlab/atom3d](https://github.com/drorlab/atom3d)

- **Paper**: Please see our [preprint](arxiv.org/XXXX) for further details on the datasets and benchmarks.

- **NeurIPS LMRL Workshop 2020**: Check out our [poster] and [talk] at the Learning Meaningful Representations of Life workshop at NeurIPS 2020.
