# Official DGL Examples and Modules

## 1998

- Page et al. [The PageRank Citation Ranking: Bringing Order to the Web](http://citeseerx.ist.psu.edu/viewdoc/summary?doi=10.1.1.38.5427).
    - Example: [[PyTorch]](../examples/pytorch/pagerank.py)
    - Tags: PageRank

## 2006

- Neuhaus et al. [Fast Suboptimal Algorithms for the Computation of Graph Edit Distance](https://link.springer.com/chapter/10.1007/11815921_17).
    - Example: [[PyTorch]](../examples/pytorch/graph_matching)
    - Tags: graph edit distance, graph matching

## 2009

- Riesen et al. [Speeding Up Graph Edit Distance Computation with a Bipartite Heuristic](https://core.ac.uk/download/pdf/33054885.pdf).
    - Example: [[PyTorch]](../examples/pytorch/graph_matching)
    - Tags: graph edit distance, graph matching

## 2011

- Fankhauser et al. [Speeding Up Graph Edit Distance Computation through Fast Bipartite Matching](https://link.springer.com/chapter/10.1007/978-3-642-20844-7_11).
    - Example: [[PyTorch]](../examples/pytorch/graph_matching)
    - Tags: graph edit distance, graph matching

- Nickel et al. [A Three-Way Model for Collective Learning on Multi-Relational Data](http://citeseerx.ist.psu.edu/viewdoc/download?doi=10.1.1.383.2015&rep=rep1&type=pdf).
    - Example: [[PyTorch]](https://github.com/awslabs/dgl-ke/tree/master/examples), [[PyTorch for custom data]](https://aws-dglke.readthedocs.io/en/latest/commands.html)
    - Tags: knowledge graph embedding
    
## 2013

- Bordes et al. [Translating Embeddings for Modeling Multi-relational Data](https://proceedings.neurips.cc/paper/2013/file/1cecc7a77928ca8133fa24680a88d2f9-Paper.pdf).
    - Example: [[PyTorch]](https://github.com/awslabs/dgl-ke/tree/master/examples), [[PyTorch for custom data]](https://aws-dglke.readthedocs.io/en/latest/commands.html)
    - Tags: knowledge graph embedding

## 2014

- Perozzi et al. [DeepWalk: Online Learning of Social Representations](https://arxiv.org/abs/1403.6652)
    - Example: [[PyTorch on OGB]](../examples/pytorch/ogb/deepwalk)
    - Tags: network embedding, transductive learning, OGB

- Fischer et al. [A Hausdorff Heuristic for Efficient Computation of Graph Edit Distance](https://link.springer.com/chapter/10.1007/978-3-662-44415-3_9).
    - Example: [[PyTorch]](../examples/pytorch/graph_matching)
    - Tags: graph edit distance, graph matching

## 2015

- Tang et al. [LINE: Large-scale Information Network Embedding](https://arxiv.org/abs/1503.03578).
    - Example: [[PyTorch on OGB]](../examples/pytorch/ogb/line)
    - Tags: network embedding, transductive learning, OGB

- Sheng Tai et al. [Improved Semantic Representations From Tree-Structured Long Short-Term Memory Networks](https://arxiv.org/abs/1503.00075).
    - Example: [PyTorch](../examples/pytorch/tree_lstm), [[MXNet]](../examples/mxnet/tree_lstm)
    - Tags: sentiment classification
    
- Vinyals et al. [Order Matters: Sequence to sequence for sets](https://arxiv.org/abs/1511.06391).
    - Pooling module: [[PyTorch]](https://docs.dgl.ai/api/python/nn.pytorch.html#set2set), [[MXNet]](https://docs.dgl.ai/api/python/nn.mxnet.html#set2set)
    - Tags: graph classification
    
- Lin et al. [Learning Entity and Relation Embeddings for Knowledge Graph Completion](https://www.aaai.org/ocs/index.php/AAAI/AAAI15/paper/viewPaper/9571).
    - Example: [[PyTorch]](https://github.com/awslabs/dgl-ke/tree/master/examples), [[PyTorch for custom data]](https://aws-dglke.readthedocs.io/en/latest/commands.html)
    - Tags: knowledge graph embedding

- Yang et al. [Embedding Entities and Relations for Learning and Inference in Knowledge Bases](https://arxiv.org/abs/1412.6575).
    - Example: [[PyTorch]](https://github.com/awslabs/dgl-ke/tree/master/examples), [[PyTorch for custom data]](https://aws-dglke.readthedocs.io/en/latest/commands.html)
    - Tags: knowledge graph embedding

## 2016

- Li et al. [Gated Graph Sequence Neural Networks](https://arxiv.org/abs/1511.05493).
    - Conv module: [[PyTorch]](https://docs.dgl.ai/api/python/nn.pytorch.html#gatedgraphconv), [TensorFlow](https://docs.dgl.ai/api/python/nn.tensorflow.html#globalattentionpooling), [[MXNet]](https://docs.dgl.ai/api/python/nn.mxnet.html#gatedgraphconv)
    - Pooling module: [[PyTorch]](https://docs.dgl.ai/api/python/nn.pytorch.html#globalattentionpooling), [[MXNet]](https://docs.dgl.ai/api/python/nn.mxnet.html#globalattentionpooling)
    - Example: [[PyTorch]](../examples/pytorch/ggnn)
    - Tags: question answering

- Defferrard et al. [Convolutional Neural Networks on Graphs with Fast Localized Spectral Filtering](https://arxiv.org/abs/1606.09375).
    - Conv module: [[PyTorch]](https://docs.dgl.ai/api/python/nn.pytorch.html#chebconv), [[PyTorch for dense graphs]](https://docs.dgl.ai/api/python/nn.pytorch.html#densechebconv), [[TensorFlow]](https://docs.dgl.ai/api/python/nn.tensorflow.html#chebconv), [[MXNet]](https://docs.dgl.ai/api/python/nn.mxnet.html#chebconv), [[MXNet for dense graphs]](https://docs.dgl.ai/api/python/nn.mxnet.html#densechebconv)
    - Example: [[PyTorch on image classification]](../examples/pytorch/model_zoo/geometric), [[PyTorch on node classification]](../examples/pytorch/model_zoo/citation_network)
    - Tags: image classification, graph classification, node classification

- Monti et al. [Geometric deep learning on graphs and manifolds using mixture model CNNs](https://arxiv.org/abs/1611.08402).
    - Conv module: [[PyTorch]](https://docs.dgl.ai/api/python/nn.pytorch.html#gmmconv), [[MXNet]](https://docs.dgl.ai/api/python/nn.mxnet.html#gmmconv)
    - Example: [[PyTorch on image classification]](../examples/pytorch/model_zoo/geometric), [PyTorch on node classification](../examples/pytorch/monet), [[MXNet on node classification]](../examples/mxnet/monet)
    - Tags: image classification, graph classification, node classification

- Kearnes et al. [Molecular Graph Convolutions: Moving Beyond Fingerprints](https://arxiv.org/abs/1603.00856).
    - Example: [PyTorch](https://github.com/awslabs/dgl-lifesci/tree/master/examples/property_prediction/moleculenet), [[PyTorch for custom data]](https://github.com/awslabs/dgl-lifesci/tree/master/examples/property_prediction/csv_data_configuration)
    - Tags: molecular property prediction

- Trouillon et al. [Complex Embeddings for Simple Link Prediction](http://proceedings.mlr.press/v48/trouillon16.pdf).
    - Example: [[PyTorch]](https://github.com/awslabs/dgl-ke/tree/master/examples), [[PyTorch for custom data]](https://aws-dglke.readthedocs.io/en/latest/commands.html)
    - Tags: knowledge graph embedding

## 2017

- Kipf and Welling. [Semi-Supervised Classification with Graph Convolutional Networks](https://arxiv.org/abs/1609.02907). 
    - Conv module: [[PyTorch]](https://docs.dgl.ai/en/latest/api/python/nn.pytorch.html#graphconv), [[PyTorch for dense graphs]](https://docs.dgl.ai/api/python/nn.pytorch.html#densegraphconv), [[TensorFlow]](https://docs.dgl.ai/en/latest/api/python/nn.tensorflow.html#graphconv), [[MXNet]](https://docs.dgl.ai/en/latest/api/python/nn.mxnet.html#dgl.nn.mxnet.conv.GraphConv), [[MXNet for dense graphs]](https://docs.dgl.ai/api/python/nn.mxnet.html#dense-conv-layers)
    - Example: [[PyTorch]](../examples/pytorch/gcn), [[PyTorch on ogbn-arxiv]](../examples/pytorch/ogb/ogbn-arxiv), [PyTorch on ogbl-ppa](https://github.com/awslabs/dgl-lifesci/tree/master/examples/link_prediction/ogbl-ppa), [PyTorch on ogbg-ppa](https://github.com/awslabs/dgl-lifesci/tree/master/examples/property_prediction/ogbg_ppa), [[TensorFlow]](../examples/tensorflow/gcn), [[MXNet]](../examples/mxnet/gcn)
    - Tags: node classification, link prediction, graph classification, OGB

- Sabour et al. [Dynamic Routing Between Capsules](https://arxiv.org/abs/1710.09829).
    - Example: [[PyTorch]](../examples/pytorch/capsule)
    - Tags: image classification
  
- van den Berg et al. [Graph Convolutional Matrix Completion](https://arxiv.org/abs/1706.02263).
    - Example: [[PyTorch]](../examples/pytorch/gcmc)
    - Tags: matrix completion, recommender system, link prediction, bipartite graphs

- Hamilton et al. [Inductive Representation Learning on Large Graphs](https://cs.stanford.edu/people/jure/pubs/graphsage-nips17.pdf).
    - Conv module: [[PyTorch]](https://docs.dgl.ai/api/python/nn.pytorch.html#sageconv), [[PyTorch for dense graphs]](https://docs.dgl.ai/api/python/nn.pytorch.html#densesageconv), [TensorFlow](https://docs.dgl.ai/api/python/nn.tensorflow.html#sageconv), [[MXNet]](https://docs.dgl.ai/api/python/nn.mxnet.html#sageconv), [[MXNet for dense graphs]](https://docs.dgl.ai/api/python/nn.mxnet.html#densesageconv)
    - Example: [[PyTorch]](../examples/pytorch/graphsage), [PyTorch on ogbn-products](../examples/pytorch/ogb/ogbn-products), [PyTorch on ogbl-ppa](https://github.com/awslabs/dgl-lifesci/tree/master/examples/link_prediction/ogbl-ppa), [[MXNet]](../examples/mxnet/graphsage)
    - Tags: node classification, sampling, unsupervised learning, link prediction

- Dong et al. [metapath2vec: Scalable Representation Learning for Heterogeneous Networks](https://dl.acm.org/doi/10.1145/3097983.3098036).
    - Example: [[PyTorch]](../examples/pytorch/metapath2vec)
    - Tags: heterogeneous graphs, network embedding, large-scale, node classification

- Du et al. [Topology Adaptive Graph Convolutional Networks](https://arxiv.org/abs/1710.10370).
    - Conv module: [[PyTorch]](https://docs.dgl.ai/api/python/nn.pytorch.html#tagconv), [[MXNet]](https://docs.dgl.ai/api/python/nn.mxnet.html#tagconv)
    - Example: [[PyTorch]](../examples/pytorch/tagcn), [[MXNet]](../examples/mxnet/tagcn)
    - Tags: node classification
    
- Qi et al. [PointNet: Deep Learning on Point Sets for 3D Classification and Segmentation](https://arxiv.org/abs/1612.00593).
    - Example: [[PyTorch]](../examples/pytorch/pointcloud/pointnet)
    - Tags: point cloud classification, point cloud part-segmentation

- Qi et al. [PointNet++: Deep Hierarchical Feature Learning on Point Sets in a Metric Space](https://arxiv.org/abs/1706.02413).
    - Example: [[PyTorch]](../examples/pytorch/pointcloud/pointnet)
    - Tags: point cloud classification
    
- Schlichtkrull. [Modeling Relational Data with Graph Convolutional Networks](https://arxiv.org/abs/1703.06103).
    - Conv module: [[PyTorch]](https://docs.dgl.ai/api/python/nn.pytorch.html#relgraphconv), [[TensorFlow]](https://docs.dgl.ai/api/python/nn.tensorflow.html#module-dgl.nn.tensorflow.conv), [[MXNet]](https://docs.dgl.ai/api/python/nn.mxnet.html#relgraphconv)
    - Example: [[PyTorch example using homogeneous DGLGraphs]](../examples/pytorch/rgcn), [[PyTorch]](../examples/pytorch/rgcn-hetero), [[TensorFlow]](../examples/tensorflow/rgcn), [[MXNet]](../examples/mxnet/rgcn)
    - Tags: node classification, link prediction, heterogeneous graphs

- Vaswani et al. [Attention Is All You Need](https://arxiv.org/abs/1706.03762).
    - Example: [[PyTorch]](../examples/pytorch/transformer)
    - Tags: machine translation

- Gilmer et al. [Neural Message Passing for Quantum Chemistry](https://arxiv.org/abs/1704.01212)
    - Conv module: [[PyTorch]](https://docs.dgl.ai/api/python/nn.pytorch.html#nnconv), [[MXNet]](https://docs.dgl.ai/api/python/nn.mxnet.html#nnconv)
    - Example: [[PyTorch]](https://github.com/awslabs/dgl-lifesci/tree/master/examples/property_prediction/alchemy), [[PyTorch for custom data]](https://github.com/awslabs/dgl-lifesci/tree/master/examples/property_prediction/csv_data_configuration)
    - Tags: molecules, quantum chemistry

- Gomes et al. [Atomic Convolutional Networks for Predicting Protein-Ligand Binding Affinity](https://arxiv.org/abs/1703.10603).
    - Conv module: [[PyTorch]](https://docs.dgl.ai/api/python/nn.pytorch.html#atomicconv)
    - Example: [[PyTorch]](https://github.com/awslabs/dgl-lifesci/tree/master/examples/binding_affinity_prediction)
    - Tags: binding affinity prediction, molecules, proteins

- Schütt et al. [SchNet: A continuous-filter convolutional neural network for modeling quantum interactions](https://arxiv.org/abs/1706.08566).
    - Conv module: [[PyTorch]](https://docs.dgl.ai/api/python/nn.pytorch.html#cfconv)
    - Example: [[PyTorch]](https://github.com/awslabs/dgl-lifesci/tree/master/examples/property_prediction/alchemy)
    - Tags: molecules, quantum chemistry

## 2018

- Li et al. [Learning Deep Generative Models of Graphs](https://arxiv.org/abs/1803.03324).
    - Example: [[PyTorch example for cycles]](../examples/pytorch/dgmg), [[PyTorch example for molecules]](https://github.com/awslabs/dgl-lifesci/tree/master/examples/generative_models/dgmg)
    - Tags: generative models, autoregressive models, molecules

- Veličković et al. [Graph Attention Networks](https://arxiv.org/abs/1710.10903).
    - Conv module: [[PyTorch]](https://docs.dgl.ai/api/python/nn.pytorch.html#gatconv), [[TensorFlow]](https://docs.dgl.ai/api/python/nn.tensorflow.html#gatconv), [[MXNet]](https://docs.dgl.ai/api/python/nn.mxnet.html#gatconv)
    - Example: [[PyTorch]](../examples/pytorch/gat), [PyTorch on ogbn-arxiv](../examples/pytorch/ogb/ogbn-arxiv), [PyTorch on ogbn-products](../examples/pytorch/ogb/ogbn-products), [[TensorFlow]](../examples/tensorflow/gat), [[MXNet]](../examples/mxnet/gat)
    - Tags: node classification, OGB

- Jin et al. [Junction Tree Variational Autoencoder for Molecular Graph Generation](https://arxiv.org/abs/1802.04364).
    - Example: [[PyTorch]](../examples/pytorch/jtnn)
    - Tags: generative models, molecules, VAE

- Thekumparampil et al. [Attention-based Graph Neural Network for Semi-supervised Learning](https://arxiv.org/abs/1803.03735).
    - Conv module: [[PyTorch]](https://docs.dgl.ai/api/python/nn.pytorch.html#agnnconv), [[MXNet]](https://docs.dgl.ai/api/python/nn.mxnet.html#agnnconv)
    - Example: [[PyTorch]](../examples/pytorch/model_zoo/citation_network)
    - Tags: node classification
    
- Ying et al. [Graph Convolutional Neural Networks for Web-Scale Recommender Systems](https://arxiv.org/abs/1806.01973).
    - Example: [[PyTorch]](../examples/pytorch/pinsage)
    - Tags: recommender system, large-scale

- Berg Palm et al. [Recurrent Relational Networks](https://arxiv.org/abs/1711.08028).
    - Example: [[PyTorch]](../examples/pytorch/rrn)
    - Tags: sudoku solving

- Yu et al. [Spatio-Temporal Graph Convolutional Networks: A Deep Learning Framework for Traffic Forecasting](https://arxiv.org/abs/1709.04875v4).
    - Example: [[PyTorch]](../examples/pytorch/stgcn_wave)
    - Tags: spatio-temporal, traffic forecasting

- Zhang et al. [An End-to-End Deep Learning Architecture for Graph Classification](https://www.cse.wustl.edu/~ychen/public/DGCNN.pdf).
    - Pooling module: [PyTorch](https://docs.dgl.ai/api/python/nn.pytorch.html#sortpooling), [TensorFlow](https://docs.dgl.ai/api/python/nn.tensorflow.html#sortpooling), [[MXNet]](https://docs.dgl.ai/api/python/nn.mxnet.html#sortpooling)
    - Tags: graph classification

## 2019

- Klicpera et al. [Predict then Propagate: Graph Neural Networks meet Personalized PageRank](https://arxiv.org/abs/1810.05997).
    - Conv module: [[PyTorch]](https://docs.dgl.ai/en/latest/api/python/nn.pytorch.html#dgl.nn.pytorch.conv.APPNPConv), [[TensorFlow]](https://docs.dgl.ai/en/latest/api/python/nn.tensorflow.html#dgl.nn.tensorflow.conv.APPNPConv), [MXNet](https://docs.dgl.ai/en/latest/api/python/nn.mxnet.html#dgl.nn.mxnet.conv.APPNPConv)
    - Example: [[PyTorch]](../examples/pytorch/appnp), [[MXNet]](../examples/mxnet/appnp)
    - Tags: node classification

- Chiang et al. [Cluster-GCN: An Efficient Algorithm for Training Deep and Large Graph Convolutional Networks](https://arxiv.org/abs/1905.07953).
    - Example: [[PyTorch]](../examples/pytorch/cluster_gcn), [[PyTorch-based GraphSAGE variant on OGB]](../examples/pytorch/ogb/cluster-sage), [[PyTorch-based GAT variant on OGB]](../examples/pytorch/ogb/cluster-gat)
    - Tags: graph partition, node classification, large-scale, OGB

- Veličković et al. [Deep Graph Infomax](https://arxiv.org/abs/1809.10341)
    - Example: [[PyTorch]](../examples/pytorch/dgi), [[TensorFlow]](../examples/tensorflow/dgi)
    - Tags: unsupervised learning

- Ying et al. [Hierarchical Graph Representation Learning with Differentiable Pooling](https://arxiv.org/abs/1806.08804)
    - Example: [[PyTorch]](../examples/pytorch/diffpool)
    - Tags: pooling, graph classification, graph coarsening

- Cen et al. [Representation Learning for Attributed Multiplex Heterogeneous Network](https://arxiv.org/abs/1905.01669v2).
    - Example: [[PyTorch]](../examples/pytorch/GATNE-T)
    - Tags: heterogeneous graphs, link prediction, large-scale

- Xu et al. [How Powerful are Graph Neural Networks?](https://arxiv.org/abs/1810.00826).
    - Conv module: [[PyTorch]](https://docs.dgl.ai/api/python/nn.pytorch.html#dgl.nn.pytorch.conv.GINConv), [[TensorFlow]](https://docs.dgl.ai/api/python/nn.tensorflow.html#ginconv), [[MXNet]](https://docs.dgl.ai/api/python/nn.mxnet.html#ginconv)
    - Example: [[PyTorch on graph classification]](../examples/pytorch/gin), [[PyTorch on node classification]](../examples/pytorch/model_zoo/citation_network), [PyTorch on ogbg-ppa](https://github.com/awslabs/dgl-lifesci/tree/master/examples/property_prediction/ogbg_ppa), [[MXNet]](../examples/mxnet/gin)
    - Tags: graph classification, node classification, OGB

- Koncel-Kedziorski et al. [Text Generation from Knowledge Graphs with Graph Transformers](https://arxiv.org/abs/1904.02342).
    - Example: [[PyTorch]](../examples/pytorch/graphwriter)
    - Tags: knowledge graph, text generation

- Wang et al. [Heterogeneous Graph Attention Network](https://arxiv.org/abs/1903.07293).
    - Example: [[PyTorch]](../examples/pytorch/han)
    - Tags: heterogeneous graphs, node classification

- Chen et al. [Supervised Community Detection with Line Graph Neural Networks](https://arxiv.org/abs/1705.08415).
    - Example: [[PyTorch]](../examples/pytorch/line_graph)
    - Tags: line graph, community detection
    
- Wu et al. [Simplifying Graph Convolutional Networks](https://arxiv.org/abs/1902.07153).
    - Conv module: [[PyTorch]](https://docs.dgl.ai/api/python/nn.pytorch.html#sgconv), [[TensorFlow]](https://docs.dgl.ai/api/python/nn.tensorflow.html#sgconv), [[MXNet]](https://docs.dgl.ai/api/python/nn.mxnet.html#sgconv)
    - Example: [[PyTorch]](../examples/pytorch/sgc), [[MXNet]](../examples/mxnet/sgc)
    - Tags: node classification

- Wang et al. [Dynamic Graph CNN for Learning on Point Clouds](https://arxiv.org/abs/1801.07829).
    - Conv module: [[PyTorch]](https://docs.dgl.ai/api/python/nn.pytorch.html#edgeconv), [[MXNet]](https://docs.dgl.ai/api/python/nn.mxnet.html#edgeconv)
    - Example: [[PyTorch]](../examples/pytorch/pointcloud/edgeconv)
    - Tags: point cloud classification

- Zhang et al. [Graphical Contrastive Losses for Scene Graph Parsing](https://arxiv.org/abs/1903.02728).
    - Example: [[MXNet]](../examples/mxnet/scenegraph)
    - Tags: scene graph extraction

- Lee et al. [Set Transformer: A Framework for Attention-based Permutation-Invariant Neural Networks](https://arxiv.org/abs/1810.00825).
    - Pooling module: [[PyTorch encoder]](https://docs.dgl.ai/api/python/nn.pytorch.html#settransformerencoder), [[PyTorch decoder]](https://docs.dgl.ai/api/python/nn.pytorch.html#settransformerdecoder)
    - Tags: graph classification

- Coley et al. [A graph-convolutional neural network model for the prediction of chemical reactivity](https://pubs.rsc.org/en/content/articlelanding/2019/sc/c8sc04228d#!divAbstract).
    - Example: [[PyTorch]](https://github.com/awslabs/dgl-lifesci/tree/master/examples/reaction_prediction/rexgen_direct)
    - Tags: molecules, reaction prediction

- Lu et al. [Molecular Property Prediction: A Multilevel Quantum Interactions Modeling Perspective](https://arxiv.org/abs/1906.11081).
    - Example: [[PyTorch]](https://github.com/awslabs/dgl-lifesci/tree/master/examples/property_prediction/alchemy)
    - Tags: molecules, quantum chemistry

- Xiong et al. [Pushing the Boundaries of Molecular Representation for Drug Discovery with the Graph Attention Mechanism](https://github.com/awslabs/dgl-lifesci/tree/master/examples/property_prediction/pubchem_aromaticity).
    - Example: [[PyTorch (with attention visualization)]](https://github.com/awslabs/dgl-lifesci/tree/master/examples/property_prediction/pubchem_aromaticity), [[PyTorch for custom data]](https://github.com/awslabs/dgl-lifesci/tree/master/examples/property_prediction/csv_data_configuration)
    - Tags: molecules, molecular property prediction

- Sun et al. [RotatE: Knowledge Graph Embedding by Relational Rotation in Complex Space](https://arxiv.org/pdf/1902.10197.pdf).
    - Example: [[PyTorch]](https://github.com/awslabs/dgl-ke/tree/master/examples), [[PyTorch for custom data]](https://aws-dglke.readthedocs.io/en/latest/commands.html)
    - Tags: knowledge graph embedding

- Abu-El-Haija et al. [MixHop: Higher-Order Graph Convolutional Architectures via Sparsified Neighborhood Mixing](https://arxiv.org/abs/1905.00067).
    - Example: [[PyTorch]](../examples/pytorch/mixhop)
    - Tags: node classification

## 2020

- Hu et al. [Heterogeneous Graph Transformer](https://arxiv.org/abs/2003.01332).
    - Example: [[PyTorch]](../examples/pytorch/hgt)
    - Tags: dynamic heterogeneous graphs, large-scale, node classification, link prediction

- Chen. [Graph Convolutional Networks for Graphs with Multi-Dimensionally Weighted Edges](https://cims.nyu.edu/~chenzh/files/GCN_with_edge_weights.pdf).
    - Example: [[PyTorch on ogbn-proteins]](../examples/pytorch/ogb/ogbn-proteins)
    - Tags: node classification, weighted graphs, OGB

- Frasca et al. [SIGN: Scalable Inception Graph Neural Networks](https://arxiv.org/abs/2004.11198).
    - Example: [[PyTorch on ogbn-arxiv/products/mag]](../examples/pytorch/ogb/sign), [[PyTorch]](../examples/pytorch/sign)
    - Tags: node classification, OGB, large-scale, heterogeneous graphs

- Hu et al. [Strategies for Pre-training Graph Neural Networks](https://arxiv.org/abs/1905.12265).
    - Example: [[Molecule embedding]](https://github.com/awslabs/dgl-lifesci/tree/master/examples/molecule_embeddings), [[PyTorch for custom data]](https://github.com/awslabs/dgl-lifesci/tree/master/examples/property_prediction/csv_data_configuration)
    - Tags: molecules, graph classification, unsupervised learning, self-supervised learning, molecular property prediction
