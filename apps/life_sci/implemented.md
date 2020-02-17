# Work Implemented in DGL-LifeSci

## Datasets/Benchmarks

- MoleculeNet: A Benchmark for Molecular Machine Learning [[paper]](https://arxiv.org/abs/1703.00564), [[website]](http://moleculenet.ai/)
    - [Tox21 with DGL](dgllife/data/tox21.py)
    - [PDBBind with DGL](dgllife/data/pdbbind.py)
- Alchemy: A Quantum Chemistry Dataset for Benchmarking AI Models [[paper]](https://arxiv.org/abs/1906.09427), [[github]](https://github.com/tencent-alchemy/Alchemy)
    - [Alchemy with DGL](dgllife/data/alchemy.py)

## Property Prediction

- Semi-Supervised Classification with Graph Convolutional Networks (GCN) [[paper]](https://arxiv.org/abs/1609.02907), [[github]](https://github.com/tkipf/gcn)
    - [GCN-Based Predictor with DGL](dgllife/model/model_zoo/gcn_predictor.py)
    - [Example for Molecule Classification](examples/property_prediction/classification.py)
- Graph Attention Networks (GAT) [[paper]](https://arxiv.org/abs/1710.10903), [[github]](https://github.com/PetarV-/GAT)
    - [GAT-Based Predictor with DGL](dgllife/model/model_zoo/gat_predictor.py)
    - [Example for Molecule Classification](examples/property_prediction/classification.py)
- SchNet: A continuous-filter convolutional neural network for modeling quantum interactions [[paper]](https://arxiv.org/abs/1706.08566), [[github]](https://github.com/atomistic-machine-learning/SchNet)
    - [SchNet with DGL](dgllife/model/model_zoo/schnet_predictor.py)
    - [Example for Molecule Regression](examples/property_prediction/regression.py)
- Molecular Property Prediction: A Multilevel Quantum Interactions Modeling Perspective (MGCN) [[paper]](https://arxiv.org/abs/1906.11081)
    - [MGCN with DGL](dgllife/model/model_zoo/mgcn_predictor.py)
    - [Example for Molecule Regression](examples/property_prediction/regression.py)
- Neural Message Passing for Quantum Chemistry (MPNN) [[paper]](https://arxiv.org/abs/1704.01212), [[github]](https://github.com/brain-research/mpnn)
    - [MPNN with DGL](dgllife/model/model_zoo/mpnn_predictor.py)
    - [Example for Molecule Regression](examples/property_prediction/regression.py)
- Pushing the Boundaries of Molecular Representation for Drug Discovery with the Graph Attention Mechanism (AttentiveFP) [[paper]](https://pubs.acs.org/doi/abs/10.1021/acs.jmedchem.9b00959)
    - [AttentiveFP with DGL](dgllife/model/model_zoo/attentivefp_predictor.py)
    - [Example for Molecule Regression](examples/property_prediction/regression.py)

## Generative Models

- Learning Deep Generative Models of Graphs (DGMG) [[paper]](https://arxiv.org/abs/1803.03324)
    - [DGMG with DGL](dgllife/model/model_zoo/dgmg.py)
    - [Example Training Script](examples/generative_models/dgmg)
- Junction Tree Variational Autoencoder for Molecular Graph Generation (JTNN) [[paper]](https://arxiv.org/abs/1802.04364)
    - [JTNN with DGL](dgllife/model/model_zoo/jtnn)
    - [Example Training Script](examples/generative_models/jtnn)

## Binding Affinity Prediction

- Atomic Convolutional Networks for Predicting Protein-Ligand Binding Affinity (ACNN) [[paper]](https://arxiv.org/abs/1703.10603), [[github]](https://github.com/deepchem/deepchem/tree/master/contrib/atomicconv)
    - [ACNN with DGL](dgllife/model/model_zoo/acnn.py)
    - [Example Training Script](examples/binding_affinity_prediction)
