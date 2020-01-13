# dglchem

## Introduction

dglchem is a DGL-based package for various applications in chemistry and drug discovery with graph neural networks. 

## Dependencies

For the time being, we only support PyTorch.

Depending on the features you want to use, you may need to manually install the following dependencies:

- RDKit 2018.09.3
    - We recommend installation with `conda install -c conda-forge rdkit==2018.09.3`. For other installation recipes,
    see the [official documentation](https://www.rdkit.org/docs/Install.html).
- MDTraj
    - We recommend installation with `conda install -c conda-forge mdtraj`. For alternative ways of installation, 
    see the [official documentation](http://mdtraj.org/1.9.3/installation.html).
