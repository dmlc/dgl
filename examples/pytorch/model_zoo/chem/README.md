# DGL for Chemistry

With atoms being nodes and bonds being edges, molecular graphs are among the core objects for study in drug discovery. 
As drug discovery is known to be costly and time consuming, deep learning on graphs can be potentially beneficial for 
improving the efficiency of drug discovery [1], [2].

With pre-trained models and training scripts, we hope this model zoo will be helpful for both
the chemistry community and the deep learning community to further their research.

## Dependencies

Before you proceed, make sure you have installed the dependencies below:
- PyTorch 1.2
    - Check the [official website](https://pytorch.org/) for installation guide
- pandas 0.24.2
    - Install with either `conda install pandas` or `pip install pandas`
- RDKit 2018.09.3
    - We recommend installation with `conda install -c conda-forge rdkit==2018.09.3`. For other installation recipes,
    see the [official documentation](https://www.rdkit.org/docs/Install.html).
- requests 2.22.0
    - Install with `pip install requests`
- scikit-learn 0.21.2
    - Install with `pip install -U scikit-learn` or `conda install scikit-learn`

## Property Prediction

[**Get started with our example code!**](https://github.com/dmlc/dgl/tree/master/examples/pytorch/model_zoo/chem/property_prediction)

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
- **SchNet**: SchNet is a novel deep learning architecture modeling quantum interactions in molecules which utilize 
the continuous-filter convolutional layers [4].   
- **Multilevel Graph Convolutional neural Network**: Multilevel Graph Convolutional neural Network (MGCN) is a well-designed 
hierarchical graph neural network directly extracts features from the conformation and spatial information followed 
by the multilevel interactions [5].    
- **Message Passing Neural Network**: Message Passing Neural Network (MPNN) is a well-designed network with edge network (enn) 
as front end and Set2Set for output prediction [6].

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
