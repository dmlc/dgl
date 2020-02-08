# DGL-LifeSci

## Introduction

Deep learning on graphs has been an arising trend in the past few years. There are a lot of graphs in 
life science such as molecular graphs and biological networks, making it an import area for applying 
deep learning on graphs. `dgllife` is a DGL-based package for various applications in life science 
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

## Organization

For a full list of work implemented in DGL-LifeSci, **see implemented.md**.

```
dgllife
    data
        csv_dataset.py
        ...
    model
        gnn
        model_zoo
        readout
        pretrain.py
    utils
        complex_to_graph.py
        early_stop.py
        eval.py
        featurizers.py
        mol_to_graph.py
        rdkit_utils.py
        splitters.py
```

### `data`

The directory consists of interfaces for working with several datasets. Additionally, one can adapt any 
`.csv` dataset to dgl with `MoleculeCSVDataset` in `csv_dataset.py`.

### `model`

- `gnn` implements several graph neural networks for message passing and updating node representations.
- `readout` implements several methods for computing graph representations out of node representations. 
In the context of molecules, they may be viewed as learned fingerprints.
- `model_zoo` implements several models for property prediction, generative models and protein-ligand 
binding affinity prediction. Many of them are based on modules in `gnn` and `readout`.
- `pretrain.py` contains APIs for loading pre-trained models.

### `utils`

- `complex_to_graph.py` contains utils for graph construction and featurization of protein-ligand complexes.
- `early_stop.py` contains utils for early stopping.
- `eval.py` contains utils for evaluating models on property prediction.
- `featurizers.py` contains utils for featurizing molecular graphs.
- `mol_to_graph.py` contains several ways for graph construction of molecules.
- `rdkit_utils.py` contains utils for RDKit, in particular loading RDKit molecule instances from different 
formats, including `mol2`, `sdf`, `pdbqt`, and `pdb`.
- `splitters.py` contains several ways for splitting the dataset.

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

| Model                      | Original Implementation | DGL Implementation | Improvement |
| -------------------------- | ----------------------- | ------------------ | ----------- |
| GCN on Tox21               | 5.5 (DeepChem)          | 1.0                | 5.5x        |
| AttentiveFP on Aromaticity | 6.0                     | 1.2                | 5x          |
| JTNN on ZINC               | 1826                    | 743                | 2.5x        |
