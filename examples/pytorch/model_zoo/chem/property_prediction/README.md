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
- **Graph Attention Networks** [7]. Graph Attention Networks (GATs) incorporate multi-head attention into GCNs,
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

| Source           | Averaged Test ROC-AUC Score |
| ---------------- | --------------------------- |
| MoleculeNet [1]  | 0.829                       |
| [DeepChem example](https://github.com/deepchem/deepchem/blob/master/examples/tox21/tox21_tensorgraph_graph_conv.py) | 0.813                  |
| Pretrained model | 0.826                       |

Note that the dataset is randomly split so these numbers are only for reference and they do not necessarily suggest
a real difference.

#### GAT on Tox21

| Source           | Averaged Test ROC-AUC Score |
| ---------------- | --------------------------- |
| Pretrained model | 0.827                       |

## Regression   

Regression tasks require assigning continuous labels to a molecule, e.g. molecular energy.

### Datasets  

- **Alchemy**. The [Alchemy Dataset](https://alchemy.tencent.com/) is introduced by Tencent Quantum Lab to facilitate the development of new 
machine learning models useful for chemistry and materials science. The dataset lists 12 quantum mechanical properties of 130,000+ organic 
molecules comprising up to 12 heavy atoms (C, N, O, S, F and Cl), sampled from the [GDBMedChem](http://gdb.unibe.ch/downloads/) database. 
These properties have been calculated using the open-source computational chemistry program Python-based Simulation of Chemistry Framework 
([PySCF](https://github.com/pyscf/pyscf)). The Alchemy dataset expands on the volume and diversity of existing molecular datasets such as QM9. 
- **PubChem BioAssay Aromaticity**. The dataset is introduced in 
[Pushing the Boundaries of Molecular Representation for Drug Discovery with the Graph Attention Mechanism](https://www.ncbi.nlm.nih.gov/pubmed/31408336), 
for the task of predicting the number of aromatic atoms in molecules. The dataset was constructed by sampling 3945 molecules with 0-40 aromatic atoms 
from the PubChem BioAssay dataset.

### Models  

- **Message Passing Neural Network** [6]. Message Passing Neural Networks (MPNNs) have reached the best performance on
the QM9 dataset for some time.
- **SchNet** [4]. SchNet employs continuous filter convolutional layers to model quantum interactions in molecules 
without requiring them to lie on grids.
- **Multilevel Graph Convolutional Neural Network** [5]. Multilevel Graph Convolutional Neural Networks (MGCN) are 
hierarchical graph neural networks that extract features from the conformation and spatial information followed by the
multilevel interactions.
- **AttentiveFP** [8]. AttentiveFP combines attention and GRU for better model capacity and shows competitive 
performance across datasetts.

### Usage

Use `regression.py` with arguments
```
-m {MPNN, SCHNET, MGCN, AttentiveFP}, Model to use
-d {Alchemy, Aromaticity}, Dataset to use
```

If you want to use the pre-trained model, simply add `-p`. Currently we only support pre-trained models of AttentiveFP
on PubChem BioAssay Aromaticity dataset.

### Performance    

#### Alchemy

The Alchemy contest is still ongoing. Before the test set is fully released, we only include the performance numbers
on the training and validation set for reference.

| Model      | Training MAE | Validation MAE |  
| ---------- | ------------ | -------------- |
| SchNet [4] | 0.2665       | 0.6139         |
| MGCN [5]   | 0.2395       | 0.6463         |
| MPNN [6]   | 0.2452       | 0.6259         |

#### PubChem BioAssay Aromaticity

| Model           | Test RMSE |
| --------------- | --------- |
| AttentiveFP [8] | 0.6998    |

## Interpretation

[8] visualizes the weights of atoms in readout for possible interpretations like the figure below. 
We provide a jupyter notebook for performing the visualization and you can download it with 
`wget https://s3.us-east-2.amazonaws.com/dgl.ai/model_zoo/drug_discovery/AttentiveFP/atom_weight_visualization.ipynb`.

![](https://s3.us-east-2.amazonaws.com/dgl.ai/model_zoo/drug_discovery/AttentiveFP/vis_example.png)

## Dataset Customization

To customize your own dataset, see the instructions
[here](https://github.com/dmlc/dgl/tree/master/python/dgl/data/chem).

## References
[1] Wu et al. (2017) MoleculeNet: a benchmark for molecular machine learning. *Chemical Science* 9, 513-530.

[2] Duvenaud et al. (2015) Convolutional networks on graphs for learning molecular fingerprints. *Advances in neural 
information processing systems (NeurIPS)*, 2224-2232.

[3] Kipf et al. (2017) Semi-Supervised Classification with Graph Convolutional Networks.
*The International Conference on Learning Representations (ICLR)*.

[4] Schütt et al. (2017) SchNet: A continuous-filter convolutional neural network for modeling quantum interactions. 
*Advances in Neural Information Processing Systems (NeurIPS)*, 992-1002.

[5] Lu et al. (2019) Molecular Property Prediction: A Multilevel Quantum Interactions Modeling Perspective. 
*The 33rd AAAI Conference on Artificial Intelligence*. 

[6] Gilmer et al. (2017) Neural Message Passing for Quantum Chemistry. *Proceedings of the 34th International Conference on 
Machine Learning*, JMLR. 1263-1272.

[7] Veličković et al. (2018) Graph Attention Networks. 
*The International Conference on Learning Representations (ICLR)*. 

[8] Xiong et al. (2019) Pushing the Boundaries of Molecular Representation for Drug Discovery with the Graph 
Attention Mechanism. *Journal of Medicinal Chemistry*.
