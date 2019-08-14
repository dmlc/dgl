# DGL for Drug Discovery

With atoms being nodes and bonds being edges, molecular graphs are among the core objects for study in drug discovery. 
As drug discovery is known to be costly and time consuming, deep learning on graphs can be potentially beneficial for 
improving the efficiency of drug discovery [1], [2].

With pre-trained models and training scripts, we hope this dgl model zoo on drug discovery will be helpful for both
the chemistry community and the deep learning community to further their research.

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
molecular graph topology, which may be viewed as a learned fingerprint [3]. For details, checkout the README of 
`gcn for property prediction`.

### Models
- **Graph Convolutional Networks**: Graph Convolutional Networks (GCN) have been one of the most popular graph neural 
network and they can be easily extended for graph level prediction.

## References

[1] Chen et al. (2018) The rise of deep learning in drug discovery. *Drug Discov Today* 6, 1241-1250.

[2] Vamathevan et al. (2019) Applications of machine learning in drug discovery and development. 
*Nature Reviews Drug Discovery* 18, 463-477.

[3] Duvenaud et al. (2015) Convolutional networks on graphs for learning molecular fingerprints. *Advances in neural 
information processing systems (NeurIPS)*, 2224-2232. 