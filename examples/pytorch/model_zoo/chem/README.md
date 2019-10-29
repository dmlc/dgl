# DGL for Chemistry

With atoms being nodes and bonds being edges, molecular graphs are among the core objects for study in Chemistry. 
Deep learning on graphs can be beneficial for various applications in Chemistry like drug and material discovery 
[1], [2], [12].

To make it easy for domain scientists, the DGL team releases a model zoo for Chemistry, focusing on two particular cases 
-- property prediction and target generation/optimization. 

With pre-trained models and training scripts, we hope this model zoo will be helpful for both
the chemistry community and the deep learning community to further their research.

## Dependencies

Before you proceed, make sure you have installed the dependencies below:
- PyTorch 1.2
    - Check the [official website](https://pytorch.org/) for installation guide.
- RDKit 2018.09.3
    - We recommend installation with `conda install -c conda-forge rdkit==2018.09.3`. For other installation recipes,
    see the [official documentation](https://www.rdkit.org/docs/Install.html).

The rest dependencies can be installed with `pip install -r requirements.txt`.

## Speed Reference

Below we provide some reference numbers to show how DGL improves the speed of training models per epoch in seconds.

| Model                      | Original Implementation | DGL Implementation | Improvement |
| -------------------------- | ----------------------- | ------------------ | ----------- |
| GCN on Tox21               | 5.5 (DeepChem)          | 1.0                | 5.5x        |
| AttentiveFP on Aromaticity | 6.0                     | 1.2                | 5x          |
| JTNN on ZINC               | 1826                    | 743                | 2.5x        |   

## Property Prediction

To evaluate molecules for drug candidates, we need to know their properties and activities. In practice, this is
mostly achieved via wet lab experiments. We can cast the problem as a regression or classification problem.
In practice, this can be quite difficult due to the scarcity of labeled data.

### Featurization and Representation Learning

Fingerprint has been a widely used concept in cheminformatics. Chemists developed hand designed rules to convert 
molecules into binary strings where each bit indicates the presence or absence of a particular substructure. The
development of fingerprints makes the comparison of molecules a lot easier. Previous machine learning methods are 
mostly developed based on molecule fingerprints.

Graph neural networks make it possible for a data-driven representation of molecules out of the atoms, bonds and 
molecular graph topology, which may be viewed as a learned fingerprint [3]. 

### Models
- **Graph Convolutional Networks** [3], [9]: Graph Convolutional Networks (GCN) have been one of the most popular graph 
neural networks and they can be easily extended for graph level prediction.
- **Graph Attention Networks** [10]: Graph Attention Networks (GATs) incorporate multi-head attention into GCNs,
explicitly modeling the interactions between adjacent atoms.
- **SchNet** [4]: SchNet is a novel deep learning architecture modeling quantum interactions in molecules which utilize 
the continuous-filter convolutional layers.   
- **Multilevel Graph Convolutional neural Network** [5]: Multilevel Graph Convolutional neural Network (MGCN) is a well-designed 
hierarchical graph neural network directly extracts features from the conformation and spatial information followed 
by the multilevel interactions.    
- **Message Passing Neural Network** [6]: Message Passing Neural Network (MPNN) is a well-designed network with edge network (enn) 
as front end and Set2Set for output prediction.

### Example Usage of Pre-trained Models

```python
from dgl.data.chem import Tox21
from dgl import model_zoo

dataset = Tox21()
model = model_zoo.chem.load_pretrained('GCN_Tox21') # Pretrained model loaded
model.eval()

smiles, g, label, mask = dataset[0]
feats = g.ndata.pop('h')
label_pred = model(feats, g)
print(smiles)                   # CCOc1ccc2nc(S(N)(=O)=O)sc2c1
print(label_pred[:, mask != 0]) # Mask non-existing labels
# tensor([[-0.7956,  0.4054,  0.4288, -0.5565, -0.0911,  
# 0.9981, -0.1663,  0.2311, -0.2376,  0.9196]])
```

## Generative Models

We use generative models for two different purposes when it comes to molecules:
- **Distribution Learning**: Given a collection of molecules, we want to model their distribution and generate new
molecules with similar properties.
- **Goal-directed Optimization**: Find molecules with desired properties.

For this model zoo, we will only focused on generative models for molecular graphs. There are other generative models 
working with alternative representations like SMILES. 

Generative models are known to be difficult for evaluation. [GuacaMol](https://github.com/BenevolentAI/guacamol) and
[MOSES](https://github.com/molecularsets/moses) have been two recent efforts to benchmark generative models. There
are also two accompanying review papers that are well written [7], [8].

### Models
- **Deep Generative Models of Graphs (DGMG)** [11]: A very general framework for graph distribution learning by 
progressively adding atoms and bonds.
- **Junction Tree Variational Autoencoder for Molecular Graph Generation (JTNN)** [13]: JTNNs are able to incrementally
expand molecules while maintaining chemical valency at every step. They can be used for both molecule generation and
optimization.

### Example Usage of Pre-trained Models

```python
# We recommend running the code below with Jupyter notebooks
from IPython.display import SVG
from rdkit import Chem
from rdkit.Chem import Draw

from dgl import model_zoo

model = model_zoo.chem.load_pretrained('DGMG_ZINC_canonical')
model.eval()
mols = []
for i in range(4):
    SMILES = model(rdkit_mol=True)
    mols.append(Chem.MolFromSmiles(SMILES))
# Generating 4 molecules takes less than a second.

SVG(Draw.MolsToGridImage(mols, molsPerRow=4, subImgSize=(180, 150), useSVG=True))
```

![](https://s3.us-east-2.amazonaws.com/dgl.ai/model_zoo/drug_discovery/dgmg_model_zoo_example2.png)

## References

[1] Chen et al. (2018) The rise of deep learning in drug discovery. *Drug Discov Today* 6, 1241-1250.

[2] Vamathevan et al. (2019) Applications of machine learning in drug discovery and development. 
*Nature Reviews Drug Discovery* 18, 463-477.

[3] Duvenaud et al. (2015) Convolutional networks on graphs for learning molecular fingerprints. *Advances in neural 
information processing systems (NeurIPS)*, 2224-2232.

[4] Schütt et al. (2017) SchNet: A continuous-filter convolutional neural network for modeling quantum interactions. 
*Advances in Neural Information Processing Systems (NeurIPS)*, 992-1002.

[5] Lu et al. Molecular Property Prediction: A Multilevel Quantum Interactions Modeling Perspective. 
*The 33rd AAAI Conference on Artificial Intelligence*. 

[6] Gilmer et al. (2017) Neural Message Passing for Quantum Chemistry. *Proceedings of the 34th International Conference on 
Machine Learning* JMLR. 1263-1272.

[7] Brown et al. (2019) GuacaMol: Benchmarking Models for de Novo Molecular Design. *J. Chem. Inf. Model*, 2019, 59, 3, 
1096-1108.

[8] Polykovskiy et al. (2019) Molecular Sets (MOSES): A Benchmarking Platform for Molecular Generation Models. *arXiv*. 

[9] Kipf et al. (2017) Semi-Supervised Classification with Graph Convolutional Networks.
*The International Conference on Learning Representations (ICLR)*. 

[10] Veličković et al. (2018) Graph Attention Networks. 
*The International Conference on Learning Representations (ICLR)*. 

[11] Li et al. (2018) Learning Deep Generative Models of Graphs. *arXiv preprint arXiv:1803.03324*.

[12] Goh et al. (2017) Deep learning for computational chemistry. *Journal of Computational Chemistry* 16, 1291-1307.

[13] Jin et al. (2018) Junction Tree Variational Autoencoder for Molecular Graph Generation. 
*Proceedings of the 35th International Conference on Machine Learning (ICML)*, 2323-2332.
