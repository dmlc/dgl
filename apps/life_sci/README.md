# DGL-LifeSci

## Introduction

Deep learning on graphs has been an arising trend in the past few years. There are a lot of graphs in 
life science such as molecular graphs and biological networks, making it an import area for applying 
deep learning on graphs. `dglls` is a DGL-based package for various applications in life science 
with graph neural networks. 

We provide various functionalities, including but not limited to methods for graph construction, 
featurization, and evaluation, model architectures, training scripts and pre-trained models.

## Dependencies

For the time being, we only support PyTorch.

Depending on the features you want to use, you may need to manually install the following dependencies:

- RDKit 2018.09.3
    - We recommend installation with `conda install -c conda-forge rdkit==2018.09.3`. For other installation recipes,
    see the [official documentation](https://www.rdkit.org/docs/Install.html).
- MDTraj
    - We recommend installation with `conda install -c conda-forge mdtraj`. For alternative ways of installation, 
    see the [official documentation](http://mdtraj.org/1.9.3/installation.html).

## Example Usage

Currently we provide examples for molecular property prediction, generative models and protein-ligand binding 
affinity prediction. See the examples folder for details.
