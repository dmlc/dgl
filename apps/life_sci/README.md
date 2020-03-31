# DGL-LifeSci

## Introduction

Deep learning on graphs has been an arising trend in the past few years. There are a lot of graphs in 
life science such as molecular graphs and biological networks, making it an import area for applying 
deep learning on graphs. DGL-LifeSci is a DGL-based package for various applications in life science 
with graph neural networks. 

We provide various functionalities, including but not limited to methods for graph construction, 
featurization, and evaluation, model architectures, training scripts and pre-trained models.

**For a full list of work implemented in DGL-LifeSci, see [here](examples/README.md).**

## Example Usage

To apply graph neural networks to molecules with DGL, we need to first construct `DGLGraph` -- 
the graph data structure in DGL and prepare initial node/edge features. Below gives an example of 
constructing a bi-directed graph from a molecule and featurizing it with atom and bond features such 
as atom type and bond type.

```python
from dgllife.utils import smiles_to_bigraph, CanonicalAtomFeaturizer, CanonicalBondFeaturizer

# Node featurizer
node_featurizer = CanonicalAtomFeaturizer(atom_data_field='h')
# Edge featurizer
edge_featurizer = CanonicalBondFeaturizer(bond_data_field='h')
# SMILES (a string representation for molecule) for Penicillin
smiles = 'CC1(C(N2C(S1)C(C2=O)NC(=O)CC3=CC=CC=C3)C(=O)O)C'
g = smiles_to_bigraph(smiles=smiles, 
                      node_featurizer=node_featurizer,
                      edge_featurizer=edge_featurizer)
print(g)
"""
DGLGraph(num_nodes=23, num_edges=50,
         ndata_schemes={'h': Scheme(shape=(74,), dtype=torch.float32)}
         edata_schemes={'h': Scheme(shape=(12,), dtype=torch.float32)})
"""
```

We implement various models that users can import directly. Below gives an example of defining a GCN-based model  
for molecular property prediction.

```python
from dgllife.model import GCNPredictor

model = GCNPredictor(in_feats=1)
```

## Dependencies

For the time being, we only support PyTorch.

Depending on the features you want to use, you may need to manually install the following dependencies:

- RDKit 2018.09.3
    - We recommend installation with `conda install -c conda-forge rdkit==2018.09.3`. For other installation recipes,
    see the [official documentation](https://www.rdkit.org/docs/Install.html).
    
## Installation

To install the package, 

```bash
cd python
python setup.py install
```

## Example Usage

Currently we provide examples for molecular property prediction, generative models and protein-ligand binding 
affinity prediction. See the examples folder for details.

For some examples we also provide pre-trained models, which can be used off-shelf without training from scratch.

```python
"""Load a pre-trained model for property prediction."""
from dgllife.data import Tox21
from dgllife.model import load_pretrained
from dgllife.utils import smiles_to_bigraph, CanonicalAtomFeaturizer

dataset = Tox21(smiles_to_bigraph, CanonicalAtomFeaturizer())
model = load_pretrained('GCN_Tox21') # Pretrained model loaded
model.eval()

smiles, g, label, mask = dataset[0]
feats = g.ndata.pop('h')
label_pred = model(g, feats)
print(smiles)                   # CCOc1ccc2nc(S(N)(=O)=O)sc2c1
print(label_pred[:, mask != 0]) # Mask non-existing labels
# tensor([[ 1.4190, -0.1820,  1.2974,  1.4416,  0.6914,  
# 2.0957,  0.5919,  0.7715, 1.7273,  0.2070]])
```

```python
"""Load a pre-trained model for generating molecules."""
from IPython.display import SVG
from rdkit import Chem
from rdkit.Chem import Draw

from dgllife.model import load_pretrained

model = load_pretrained('DGMG_ZINC_canonical')
model.eval()
mols = []
for i in range(4):
    SMILES = model(rdkit_mol=True)
    mols.append(Chem.MolFromSmiles(SMILES))
# Generating 4 molecules takes less than a second.

SVG(Draw.MolsToGridImage(mols, molsPerRow=4, subImgSize=(180, 150), useSVG=True))
```

![](https://data.dgl.ai/dgllife/dgmg/dgmg_model_zoo_example2.png)

## Speed Reference

Below we provide some reference numbers to show how DGL improves the speed of training models per epoch in seconds.

| Model                              | Original Implementation | DGL Implementation | Improvement |
| ---------------------------------- | ----------------------- | ------------------ | ----------- |
| GCN on Tox21                       | 5.5 (DeepChem)          | 1.0                | 5.5x        |
| AttentiveFP on Aromaticity         | 6.0                     | 1.2                | 5x          |
| JTNN on ZINC                       | 1826                    | 743                | 2.5x        |
| WLN for reaction center prediction | 11657                   | 5095               | 2.3x        |                                                           |
