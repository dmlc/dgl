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
- **Graph Convolutional Network** [2]. Graph Convolutional Networks (GCN) have been one of the most popular graph neural
networks and they can be easily extended for graph level prediction. MoleculeNet [1] reports baseline results of graph
convolutions over multiple datasets.

### Usage

To train a model from scratch, simply call `python classification.py`. To skip training and use the pre-trained model,
call `python classification.py -p`.

We use GPU whenever it is available.

### Performance

#### GCN on Tox21

| Source           | Averaged ROC-AUC Score |
| ---------------- | ---------------------- |
| MoleculeNet [1]  | 0.829                  |
| [DeepChem example](https://github.com/deepchem/deepchem/blob/master/examples/tox21/tox21_tensorgraph_graph_conv.py) | 0.813                  |
| Pretrained model | 0.827                  |

Note that due to some possible randomness you may get different numbers for DeepChem example and our model. To get
match exact results for this model, please use the pre-trained model as in the usage section.

## Dataset Customization

To customize your own dataset, see the instructions
[here](https://github.com/dmlc/dgl/tree/master/python/dgl/data/chem).

## Regression   

Regression tasks require assigning continuous labels to a molecule, e.g. molecular energy.

###Dataset  

- **Alchemy**. The [Alchemy Dataset](https://alchemy.tencent.com/) is introduced by Tencent Quantum Lab to facilitate the development of new machine learning models useful for chemistry and materials science. 
The dataset lists 12 quantum mechanical properties of 130,000+ organic molecules comprising up to 12 heavy atoms (C, N, O, S, F and Cl), sampled from the [GDBMedChem](http://gdb.unibe.ch/downloads/) database. 
These properties have been calculated using the open-source computational chemistry program Python-based Simulation of Chemistry Framework ([PySCF](https://github.com/pyscf/pyscf)). 
The Alchemy dataset expands on the volume and diversity of existing molecular datasets such as QM9.  

### Models  

- **SchNet**: SchNet is a novel deep learning architecture modeling quantum interactions in molecules which utilize the continuous-filter convolutional layers [3].   
- **Multilevel Graph Convolutional neural Network**: Multilevel Graph Convolutional neural Network (MGCN) is a well-designed hierarchical graph neural network directly extracts features from the conformation and spatial information followed by the multilevel interactions [4].    
- **Message Passing Neural Network**: Message Passing Neural Network (MPNN) is a well-designed network with edge network (enn) as front end and us Set2Set to output prediction [5].

### Usage

```py  
python regression.py --model sch --epoch 200
```  
The model option must be one of 'sch', 'mgcn' or 'mpnn'.  

### Performance    

#### Alchemy   

|Model        |Mean Absolute Error (MAE)|  
|-------------|-------------------------|
|SchNet[3]    |0.065|
|MGCN[4]      |0.050|
|MPNN[5]      |0.056|

## References
[1] Wu et al. (2017) MoleculeNet: a benchmark for molecular machine learning. *Chemical Science* 9, 513-530.

[2] Kipf et al. (2017) Semi-Supervised Classification with Graph Convolutional Networks.
*The International Conference on Learning Representations (ICLR)*.

[3] Schütt et al. (2017) SchNet: A continuous-filter convolutional neural network for modeling quantum interactions. 
*Advances in Neural Information Processing Systems (NeurIPS)*, 992-1002.

[4] Lu et al. (2019) Molecular Property Prediction: A Multilevel Quantum Interactions Modeling Perspective. 
*The 33rd AAAI Conference on Artificial Intelligence*. 

[5] Gilmer et al. (2017) Neural Message Passing for Quantum Chemistry. *Proceedings of the 34th International Conference on 
Machine Learning*, JMLR. 1263-1272.

