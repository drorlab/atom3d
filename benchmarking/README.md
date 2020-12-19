# Benchmarking for the ATOM3D paper

This directory contains scripts we used for the benchmarking described in the [ATOM3D paper](https://arxiv.org/abs/2012.04035) and corresponding (though incomplete) documentation.
Parts of the code here rely on datasets in an outdated format that we can only access locally. We are trying to update them over time to make them more usable. 


## Network Architectures

We benchmark the following prototypical architectures:
* 3D-CNNs: three-dimensional convolutional neural networks (based on SASnet)  
* GNNs: graph neural networks (based on pytorch-geometric)
* ENNs: equivariant neural networks (Cormorant)

See the README files in the corresponding folders for details.

## Reference

Benchmarking results can be found in our preprint:

> R. J. L. Townshend, M. Vögele, P. Suriana, A. Derry, A. Powers, Y. Laloudakis, S. Balachandar, B. Anderson, S. Eismann, R. Kondor, R. B. Altman, R. O. Dror "ATOM3D: Tasks On Molecules in Three Dimensions", [arXiv:2012.04035](https://arxiv.org/abs/2012.04035)
  
Please cite this work if some of the ATOM3D code or datasets are helpful in your scientific endeavours. For specific datasets or model implementations, please also cite the respective original source(s), given in the preprint.


