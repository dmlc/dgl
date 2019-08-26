# Property Prediction

## Classification

Classification tasks require assigning discrete labels to a molecule, e.g. molecule toxicity.

### Datasets
- **Tox21**. The ["Toxicology in the 21st Century" (Tox21)](https://tripod.nih.gov/tox21/challenge/) initiative created
a public database measuring toxicity of compounds, which has been used in the 2014 Tox21 Data Challenge. The dataset
contains qualitative toxicity measurements for 8014 compounds on 12 different targets, including nuclear receptors and
stress response pathways. Each target yields a binary prediction problem. MoleculeNet [1] randomly splits the dataset
into training, validation and test set with a 80/10/10 ratio. By default we follow their split method.

### Models
- **Graph Convolutional Network** [2], [3]. Graph Convolutional Networks (GCN) have been one of the most popular graph neural
networks and they can be easily extended for graph level prediction. MoleculeNet [1] reports baseline results of graph
convolutions over multiple datasets.
- **Graph Attention Networks** [4]: Graph Attention Networks (GATs) incorporate multi-head attention into GCNs,
explicitly modeling the interactions between adjacent atoms.

### Usage

Use `classification.py` with arguments
```
-m {GCN, GAT}, MODEL, model to use
-d {Tox21}, DATASET, dataset to use
```

If you want to use the pre-trained model, simply add `-p`.

We use GPU whenever it is available.

### Performance

#### GCN on Tox21

| Source           | Averaged ROC-AUC Score |
| ---------------- | ---------------------- |
| MoleculeNet [1]  | 0.829                  |
| [DeepChem example](https://github.com/deepchem/deepchem/blob/master/examples/tox21/tox21_tensorgraph_graph_conv.py) | 0.813                  |
| Pretrained model | 0.826                  |

Note that the dataset is randomly split so these numbers are only for reference and they do not necessarily suggest
a real difference.

#### GAT on Tox21

| Source           | Averaged ROC-AUC Score |
| ---------------- | ---------------------- |
| Pretrained model | 0.827                  |

## Dataset Customization

To customize your own dataset, see the instructions
[here](https://github.com/dmlc/dgl/tree/master/python/dgl/data/chem).

### References
[1] Wu et al. (2017) MoleculeNet: a benchmark for molecular machine learning. *Chemical Science* 9, 513-530.

[2] Duvenaud et al. (2015) Convolutional networks on graphs for learning molecular fingerprints. *Advances in neural 
information processing systems (NeurIPS)*, 2224-2232.

[3] Kipf et al. (2017) Semi-Supervised Classification with Graph Convolutional Networks.
*The International Conference on Learning Representations (ICLR)*.

[4] Veličković et al. (2018) Graph Attention Networks. 
*The International Conference on Learning Representations (ICLR)*. 
