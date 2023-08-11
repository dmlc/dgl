<p align="center">
  <img src="http://data.dgl.ai/asset/logo.jpg" height="200">
</p>

[![Latest Release](https://img.shields.io/github/v/release/dmlc/dgl)](https://github.com/dmlc/dgl/releases)
[![Conda Latest Release](https://anaconda.org/dglteam/dgl/badges/version.svg)](https://anaconda.org/dglteam/dgl)
[![Build Status](https://ci.dgl.ai/buildStatus/icon?job=DGL/master)](https://ci.dgl.ai/job/DGL/job/master/)
[![Benchmark by ASV](http://img.shields.io/badge/benchmarked%20by-asv-green.svg?style=flat)](https://asv.dgl.ai/)
[![License](https://img.shields.io/badge/License-Apache%202.0-blue.svg)](./LICENSE)
[![Twitter](https://img.shields.io/twitter/follow/DGLGraph?style=social)](https://twitter.com/GraphDeep)

[Website](https://www.dgl.ai) | [A Blitz Introduction to DGL](https://docs.dgl.ai/tutorials/blitz/index.html) | Documentation ([Latest](https://docs.dgl.ai/en/latest/) | [Stable](https://docs.dgl.ai)) | [Official Examples](examples/README.md) | [Discussion Forum](https://discuss.dgl.ai) | [Slack Channel](https://join.slack.com/t/deep-graph-library/shared_invite/zt-eb4ict1g-xcg3PhZAFAB8p6dtKuP6xQ)

DGL is an easy-to-use, high performance and scalable Python package for deep learning on graphs. DGL is framework agnostic, meaning if a deep graph model is a component of an end-to-end application, the rest of the logics can be implemented in any major frameworks, such as PyTorch, Apache MXNet or TensorFlow.

<p align="center">
  <img src="http://data.dgl.ai/asset/image/DGL-Arch.png" alt="DGL v0.4 architecture" width="600">
  <br>
  <b>Figure</b>: DGL Overall Architecture
</p>

## Highlighted Features

### A GPU-ready graph library

DGL provides a powerful graph object that can reside on either CPU or GPU. It bundles structural data as well as features for better control. We provide a variety of functions for computing with graph objects including efficient and customizable message passing primitives for Graph Neural Networks.

### A versatile tool for GNN researchers and practitioners

The field of graph deep learning is still rapidly evolving and many research ideas emerge by standing on the shoulders of giants. To ease the process, [DGl-Go](https://github.com/dmlc/dgl/tree/master/dglgo) is a command-line interface to get started with training, using and studying state-of-the-art GNNs.
DGL collects a rich set of [example implementations](https://github.com/dmlc/dgl/tree/master/examples) of popular GNN models of a wide range of topics. Researchers can [search](https://www.dgl.ai/) for related models to innovate new ideas from or use them as baselines for experiments. Moreover, DGL provides many state-of-the-art [GNN layers and modules](https://docs.dgl.ai/api/python/nn.html) for users to build new model architectures. DGL is one of the preferred platforms for many standard graph deep learning benchmarks including [OGB](https://ogb.stanford.edu/) and [GNNBenchmarks](https://github.com/graphdeeplearning/benchmarking-gnns).

### Easy to learn and use

DGL provides plenty of learning materials for all kinds of users from ML researchers to domain experts. The [Blitz Introduction to DGL](https://docs.dgl.ai/tutorials/blitz/index.html) is a 120-minute tour of the basics of graph machine learning. The [User Guide](https://docs.dgl.ai/guide/index.html) explains in more details the concepts of graphs as well as the training methodology. All of them include code snippets in DGL that are runnable and ready to be plugged into one’s own pipeline.

### Scalable and efficient

It is convenient to train models using DGL on large-scale graphs across **multiple GPUs** or **multiple machines**. DGL extensively optimizes the whole stack to reduce the overhead in communication, memory consumption and synchronization. As a result, DGL can easily scale to billion-sized graphs. Get started with the [tutorials](https://docs.dgl.ai/en/tutorials/dist/index.html) and [user guide](https://docs.dgl.ai/en/latest/guide/distributed.html) for distributed training. See the [system performance note](https://docs.dgl.ai/performance.html) for the comparison with other tools.

## Get Started

Users can install DGL from [pip and conda](https://www.dgl.ai/pages/start.html). You can also download GPU enabled DGL docker [containers](https://catalog.ngc.nvidia.com/orgs/nvidia/containers/dgl) (backended by PyTorch) from NVIDIA NGC for both x86 and ARM based linux systems. Advanced users can follow the [instructions](https://docs.dgl.ai/install/index.html#install-from-source) to install from source.

For absolute beginners, start with [the Blitz Introduction to DGL](https://docs.dgl.ai/tutorials/blitz/index.html). It covers the basic concepts of common graph machine learning tasks and a step-by-step on building Graph Neural Networks (GNNs) to solve them.

For acquainted users who wish to learn more,

* Experience state-of-the-art GNN models in only two command-lines using [DGL-Go](https://github.com/dmlc/dgl/tree/master/dglgo).
* Learn DGL by [example implementations](https://www.dgl.ai/) of popular GNN models.
* Read the [User Guide](https://docs.dgl.ai/guide/index.html) ([中文版链接](https://docs.dgl.ai/guide_cn/index.html)), which explains the concepts and usage of DGL in much more details.
* Go through the tutorials for advanced features like [stochastic training of GNNs](https://docs.dgl.ai/tutorials/large/index.html), training on [multi-GPU](https://docs.dgl.ai/tutorials/multi/index.html) or [multi-machine](https://docs.dgl.ai/tutorials/dist/index.html).
* [Study classical papers](https://docs.dgl.ai/tutorials/models/index.html) on graph machine learning alongside DGL.
* Search for the usage of a specific API in the [API reference manual](https://docs.dgl.ai/api/python/index.html), which organizes all DGL APIs by their namespace.

All the learning materials are available at our [documentation site](https://docs.dgl.ai/). If you are new to deep learning in general,
check out the open source book [Dive into Deep Learning](https://d2l.ai/).


## Community

### Get connected

We provide multiple channels to connect you to the community of the DGL developers, users, and the general GNN academic researchers:

* Our Slack channel, [click to join](https://join.slack.com/t/deep-graph-library/shared_invite/zt-eb4ict1g-xcg3PhZAFAB8p6dtKuP6xQ)
* Our discussion forum: https://discuss.dgl.ai/
* Our [Zhihu blog (in Chinese)](https://www.zhihu.com/column/c_1070749881013936128)
* Monthly GNN User Group online seminar ([event link](https://www.eventbrite.com/e/graph-neural-networks-user-group-tickets-137512275919?utm-medium=discovery&utm-campaign=social&utm-content=attendeeshare&aff=escb&utm-source=cp&utm-term=listing) | [past videos](https://www.youtube.com/channel/UCnmuSDY1pTlaFH1WRQElfTg))

Take the survey [here](https://forms.gle/Ej3jHCocACmb49Gp8) and leave any feedback to make DGL better fit for your needs. Thanks!

### DGL-powered projects

* DGL-LifeSci: a DGL-based package for various applications in life science with graph neural networks. https://github.com/awslabs/dgl-lifesci
* DGL-KE: a high performance, easy-to-use, and scalable package for learning large-scale knowledge graph embeddings. https://github.com/awslabs/dgl-ke
* Benchmarking GNN: https://github.com/graphdeeplearning/benchmarking-gnns
* OGB: a collection of realistic, large-scale, and diverse benchmark datasets for machine learning on graphs. https://ogb.stanford.edu/
* Graph4NLP: an easy-to-use library for R&D at the intersection of Deep Learning on Graphs and Natural Language Processing. https://github.com/graph4ai/graph4nlp
* GNN-RecSys: https://github.com/je-dbl/GNN-RecSys
* Amazon Neptune ML: a new capability of Neptune that uses Graph Neural Networks (GNNs), a machine learning technique purpose-built for graphs, to make easy, fast, and more accurate predictions using graph data. https://aws.amazon.com/cn/neptune/machine-learning/
* GNNLens2: Visualization tool for Graph Neural Networks. https://github.com/dmlc/GNNLens2
* RNAGlib: A package to facilitate construction, analysis, visualization and machine learning on RNA 2.5D Graphs. Includes a pre-built dataset: https://rnaglib.cs.mcgill.ca
* OpenHGNN: Model zoo and benchmarks for Heterogeneous Graph Neural Networks. https://github.com/BUPT-GAMMA/OpenHGNN
* TGL: A graph learning framework for large-scale temporal graphs. https://github.com/amazon-research/tgl
* gtrick: Bag of Tricks for Graph Neural Networks. https://github.com/sangyx/gtrick
* ArangoDB-DGL Adapter: Import [ArangoDB](https://github.com/arangodb/arangodb) graphs into DGL and vice-versa. https://github.com/arangoml/dgl-adapter
* DGLD: [DGLD](https://github.com/EagleLab-ZJU/DGLD) is an open-source library for Deep Graph Anomaly Detection based on pytorch and DGL.
### Awesome Papers Using DGL

1. [**Benchmarking Graph Neural Networks**](https://arxiv.org/pdf/2003.00982.pdf), *Vijay Prakash Dwivedi, Chaitanya K. Joshi, Thomas Laurent, Yoshua Bengio, Xavier Bresson*

1. [**Open Graph Benchmarks: Datasets for Machine Learning on Graphs**](https://arxiv.org/pdf/2005.00687.pdf), NeurIPS'20, *Weihua Hu, Matthias Fey, Marinka Zitnik, Yuxiao Dong, Hongyu Ren, Bowen Liu, Michele Catasta, Jure Leskovec*

1. [**DropEdge: Towards Deep Graph Convolutional Networks on Node Classification**](https://openreview.net/pdf?id=Hkx1qkrKPr), ICLR'20, *Yu Rong, Wenbing Huang, Tingyang Xu, Junzhou Huan*

1. [**Discourse-Aware Neural Extractive Text Summarization**](https://www.aclweb.org/anthology/2020.acl-main.451/), ACL'20, *Jiacheng Xu, Zhe Gan, Yu Cheng, Jingjing Liu*

1. [**GCC: Graph Contrastive Coding for Graph Neural Network Pre-Training**](https://dl.acm.org/doi/pdf/10.1145/3394486.3403168?casa_token=EClsH2Vc4DcAAAAA:LIB8cbtr6yTDbYuv4cTLwTIYeDq5Y2dhj_ktcWdKpzdPLGeiuL0o8GlcN4QIOnpsAnmGeGVZ), KDD'20, *Jiezhong Qiu, Qibin Chen, Yuxiao Dong, Jing Zhang, Hongxia Yang, Ming Ding, Kuansan Wang, Jie Tang*

1. [**DGL-KE: Training Knowledge Graph Embeddings at Scale**](https://arxiv.org/pdf/2004.08532), SIGIR'20, *Da Zheng, Xiang Song, Chao Ma, Zeyuan Tan, Zihao Ye, Jin Dong, Hao Xiong, Zheng Zhang, George Karypis*

1. [**Improving Graph Neural Network Expressivity via Subgraph Isomorphism Counting**](https://arxiv.org/pdf/2006.09252.pdf), *Giorgos Bouritsas, Fabrizio Frasca, Stefanos Zafeiriou, Michael M. Bronstein*

1. [**INT: An Inequality Benchmark for Evaluating Generalization in Theorem Proving**](https://arxiv.org/pdf/2007.02924.pdf), *Yuhuai Wu, Albert Q. Jiang, Jimmy Ba, Roger Grosse*

1. [**Finding Patient Zero: Learning Contagion Source with Graph Neural Networks**](https://arxiv.org/pdf/2006.11913.pdf), *Chintan Shah, Nima Dehmamy, Nicola Perra, Matteo Chinazzi, Albert-László Barabási, Alessandro Vespignani, Rose Yu*

1. [**FeatGraph: A Flexible and Efficient Backend for Graph Neural Network Systems**](https://arxiv.org/pdf/2008.11359.pdf), SC'20, *Yuwei Hu, Zihao Ye, Minjie Wang, Jiali Yu, Da Zheng, Mu Li, Zheng Zhang, Zhiru Zhang, Yida Wang*


<details><summary>more</summary>

11. [**BP-Transformer: Modelling Long-Range Context via Binary Partitioning.**](https://arxiv.org/pdf/1911.04070.pdf), *Zihao Ye, Qipeng Guo, Quan Gan, Xipeng Qiu, Zheng Zhang*

12. [**OptiMol: Optimization of Binding Affinities in Chemical Space for Drug Discovery**](https://www.biorxiv.org/content/biorxiv/early/2020/06/16/2020.05.23.112201.full.pdf), *Jacques Boitreaud,Vincent Mallet, Carlos Oliver, Jérôme Waldispühl*

1. [**JAKET: Joint Pre-training of Knowledge Graph and Language Understanding**](https://arxiv.org/pdf/2010.00796.pdf), *Donghan Yu, Chenguang Zhu, Yiming Yang, Michael Zeng*

1. [**Architectural Implications of Graph Neural Networks**](https://arxiv.org/pdf/2009.00804.pdf), *Zhihui Zhang, Jingwen Leng, Lingxiao Ma, Youshan Miao, Chao Li, Minyi Guo*

1. [**Combining Reinforcement Learning and Constraint Programming for Combinatorial Optimization**](https://arxiv.org/pdf/2006.01610.pdf), *Quentin Cappart, Thierry Moisan, Louis-Martin Rousseau1, Isabeau Prémont-Schwarz, and Andre Cire*

1. [**Therapeutics Data Commons: Machine Learning Datasets and Tasks for Therapeutics**](https://arxiv.org/abs/2102.09548) ([code repo](https://github.com/mims-harvard/TDC)), *Kexin Huang, Tianfan Fu, Wenhao Gao, Yue Zhao, Yusuf Roohani, Jure Leskovec, Connor W. Coley, Cao Xiao, Jimeng Sun, Marinka Zitnik*

1. [**Sparse Graph Attention Networks**](https://arxiv.org/abs/1912.00552), *Yang Ye, Shihao Ji*

1. [**On Self-Distilling Graph Neural Network**](https://arxiv.org/pdf/2011.02255.pdf), *Yuzhao Chen, Yatao Bian, Xi Xiao, Yu Rong, Tingyang Xu, Junzhou Huang*

1. [**Learning Robust Node Representations on Graphs**](https://arxiv.org/pdf/2008.11416.pdf), *Xu Chen, Ya Zhang, Ivor Tsang, and Yuangang Pan*

1. [**Recurrent Event Network: Autoregressive Structure Inference over Temporal Knowledge Graphs**](https://arxiv.org/abs/1904.05530), *Woojeong Jin, Meng Qu, Xisen Jin, Xiang Ren*

1. [**Graph Neural Ordinary Differential Equations**](https://arxiv.org/abs/1911.07532), *Michael Poli, Stefano Massaroli, Junyoung Park, Atsushi Yamashita, Hajime Asama, Jinkyoo Park*

1. [**FusedMM: A Unified SDDMM-SpMM Kernel for Graph Embedding and Graph Neural Networks**](https://arxiv.org/pdf/2011.06391.pdf), *Md. Khaledur Rahman, Majedul Haque Sujon, , Ariful Azad*

1. [**An Efficient Neighborhood-based Interaction Model for Recommendation on Heterogeneous Graph**](https://arxiv.org/pdf/2007.00216.pdf), KDD'20 *Jiarui Jin, Jiarui Qin, Yuchen Fang, Kounianhua Du, Weinan Zhang, Yong Yu, Zheng Zhang, Alexander J. Smola*

1. [**Learning Interaction Models of Structured Neighborhood on Heterogeneous Information Network**](https://arxiv.org/pdf/2011.12683.pdf), *Jiarui Jin, Kounianhua Du, Weinan Zhang, Jiarui Qin, Yuchen Fang, Yong Yu, Zheng Zhang, Alexander J. Smola*

1. [**Graphein - a Python Library for Geometric Deep Learning and Network Analysis on Protein Structures**](https://www.biorxiv.org/content/10.1101/2020.07.15.204701v1), *Arian R. Jamasb, Pietro Lió, Tom L. Blundell*

1. [**Graph Policy Gradients for Large Scale Robot Control**](https://arxiv.org/abs/1907.03822), *Arbaaz Khan, Ekaterina Tolstaya, Alejandro Ribeiro, Vijay Kumar*

1. [**Heterogeneous Molecular Graph Neural Networks for Predicting Molecule Properties**](https://arxiv.org/abs/2009.12710), *Zeren Shui, George Karypis*

1. [**Could Graph Neural Networks Learn Better Molecular Representation for Drug Discovery? A Comparison Study of Descriptor-based and Graph-based Models**](https://assets.researchsquare.com/files/rs-81439/v1_stamped.pdf), *Dejun Jiang, Zhenxing Wu, Chang-Yu Hsieh, Guangyong Chen, Ben Liao, Zhe Wang, Chao Shen, Dongsheng Cao, Jian Wu, Tingjun Hou*

1. [**Principal Neighbourhood Aggregation for Graph Nets**](https://arxiv.org/abs/2004.05718), *Gabriele Corso, Luca Cavalleri, Dominique Beaini, Pietro Liò, Petar Veličković*

1. [**Collective Multi-type Entity Alignment Between Knowledge Graphs**](https://dl.acm.org/doi/abs/10.1145/3366423.3380289), *Qi Zhu, Hao Wei, Bunyamin Sisman, Da Zheng, Christos Faloutsos, Xin Luna Dong, Jiawei Han*

1. [**Graph Representation Forecasting of Patient's Medical Conditions: towards A Digital Twin**](https://arxiv.org/abs/2009.08299), *Pietro Barbiero, Ramon Viñas Torné, Pietro Lió*

1. [**Relational Graph Learning on Visual and Kinematics Embeddings for Accurate Gesture Recognition in Robotic Surgery**](https://arxiv.org/abs/2011.01619), *Yong-Hao Long, Jie-Ying Wu, Bo Lu, Yue-Ming Jin, Mathias Unberath, Yun-Hui Liu, Pheng-Ann Heng and Qi Dou*

1. [**Dark Reciprocal-Rank: Boosting Graph-Convolutional Self-Localization Network via Teacher-to-student Knowledge Transfer**](https://arxiv.org/abs/2011.00402), *Takeda Koji, Tanaka Kanji*

1. [**Graph InfoClust: Leveraging Cluster-Level Node Information For Unsupervised Graph Representation Learning**](https://arxiv.org/abs/2009.06946), *Costas Mavromatis, George Karypis*

1. [**GraphSeam: Supervised Graph Learning Framework for Semantic UV Mapping**](https://arxiv.org/abs/2011.13748), *Fatemeh Teimury, Bruno Roy, Juan Sebastian Casallas, David macdonald, Mark Coates*

1. [**Comprehensive Study on Molecular Supervised Learning with Graph Neural Networks**](https://pubs.acs.org/doi/10.1021/acs.jcim.0c00416), *Doyeong Hwang, Soojung Yang, Yongchan Kwon, Kyung Hoon Lee, Grace Lee, Hanseok Jo, Seyeol Yoon, and Seongok Ryu*

1. [**A graph auto-encoder model for miRNA-disease associations prediction**](https://academic.oup.com/bib/advance-article-abstract/doi/10.1093/bib/bbaa240/5929824?redirectedFrom=fulltext), *Zhengwei Li, Jiashu Li, Ru Nie, Zhu-Hong You, Wenzheng Bao*

1. [**Graph convolutional regression of cardiac depolarization from sparse endocardial maps**](https://arxiv.org/abs/2009.14068), STACOM 2020 workshop, *Felix Meister, Tiziano Passerini, Chloé Audigier, Èric Lluch, Viorel Mihalef, Hiroshi Ashikaga, Andreas Maier, Henry Halperin, Tommaso Mansi*

1. [**AttnIO: Knowledge Graph Exploration with In-and-Out Attention Flow for Knowledge-Grounded Dialogue**](https://www.aclweb.org/anthology/2020.emnlp-main.280/), EMNLP'20, *Jaehun Jung, Bokyung Son, Sungwon Lyu*

1. [**Learning from Non-Binary Constituency Trees via Tensor Decomposition**](https://github.com/danielecastellana22/tensor-tree-nn), COLING'20, *Daniele Castellana, Davide Bacciu*

1. [**Inducing Alignment Structure with Gated Graph Attention Networks for Sentence Matching**](https://arxiv.org/abs/2010.07668), *Peng Cui, Le Hu, Yuanchao Liu*

1. [**Enhancing Extractive Text Summarization with Topic-Aware Graph Neural Networks**](https://arxiv.org/abs/2010.06253), COLING'20, *Peng Cui, Le Hu, Yuanchao Liu*

1. [**Double Graph Based Reasoning for Document-level Relation Extraction**](https://arxiv.org/abs/2009.13752), EMNLP'20, *Shuang Zeng, Runxin Xu, Baobao Chang, Lei Li*

1. [**Systematic Generalization on gSCAN with Language Conditioned Embedding**](https://arxiv.org/abs/2009.05552), AACL-IJCNLP'20, *Tong Gao, Qi Huang, Raymond J. Mooney*

1. [**Automatic selection of clustering algorithms using supervised graph embedding**](https://arxiv.org/pdf/2011.08225.pdf), *Noy Cohen-Shapira, Lior Rokach*

1. [**Improving Learning to Branch via Reinforcement Learning**](https://openreview.net/forum?id=z4D7-PTxTb), *Haoran Sun, Wenbo Chen, Hui Li, Le Song*

1. [**A Practical Guide to Graph Neural Networks**](https://arxiv.org/pdf/2010.05234.pdf), *Isaac Ronald Ward, Jack Joyner, Casey Lickfold, Stash Rowe, Yulan Guo, Mohammed Bennamoun*, [code](https://github.com/isolabs/gnn-tutorial)

1. [**APAN: Asynchronous Propagation Attention Network for Real-time Temporal Graph Embedding**](https://arxiv.org/pdf/2011.11545.pdf), SIGMOD'21, *Xuhong Wang, Ding Lyu, Mengjian Li, Yang Xia, Qi Yang, Xinwen Wang, Xinguang Wang, Ping Cui, Yupu Yang, Bowen Sun, Zhenyu Guo, Junkui Li*

1. [**Uncertainty-Matching Graph Neural Networks to Defend Against Poisoning Attacks**](https://arxiv.org/pdf/2009.14455.pdf), *Uday Shankar Shanthamallu, Jayaraman J. Thiagarajan, Andreas Spanias*

1. [**Computing Graph Neural Networks: A Survey from Algorithms to Accelerators**](https://arxiv.org/pdf/2010.00130.pdf), *Sergi Abadal, Akshay Jain, Robert Guirado, Jorge López-Alonso, Eduard Alarcón*

1. [**NHK_STRL at WNUT-2020 Task 2: GATs with Syntactic Dependencies as Edges and CTC-based Loss for Text Classification**](https://www.aclweb.org/anthology/2020.wnut-1.43.pdf), *Yuki Yasuda, Taichi Ishiwatari, Taro Miyazaki, Jun Goto*

1. [**Relation-aware Graph Attention Networks with Relational Position Encodings for Emotion Recognition in Conversations**](https://www.aclweb.org/anthology/2020.emnlp-main.597.pdf), *Taichi Ishiwatari, Yuki Yasuda, Taro Miyazaki, Jun Goto*

1. [**PGM-Explainer: Probabilistic Graphical Model Explanations for Graph Neural Networks**](https://proceedings.neurips.cc/paper/2020/file/8fb134f258b1f7865a6ab2d935a897c9-Paper.pdf), *Minh N. Vu, My T. Thai*

1. [**A Generalization of Transformer Networks to Graphs**](https://arxiv.org/pdf/2012.09699.pdf), *Vijay Prakash Dwivedi, Xavier Bresson*

1. [**Discourse-Aware Neural Extractive Text Summarization**](https://www.aclweb.org/anthology/2020.acl-main.451.pdf), ACL'20, *Jiacheng Xu, Zhe Gan, Yu Cheng, Jingjing Liu*

1. [**Learning Robust Node Representations on Graphs**](https://arxiv.org/abs/2008.11416), *Xu Chen, Ya Zhang, Ivor Tsang, Yuangang Pan*

1. [**Adaptive Graph Diffusion Networks with Hop-wise Attention**](https://arxiv.org/abs/2012.15024), *Chuxiong Sun, Guoshi Wu*

1. [**The Photoswitch Dataset: A Molecular Machine Learning Benchmark for the Advancement of Synthetic Chemistry**](https://arxiv.org/abs/2008.03226), *Aditya R. Thawani, Ryan-Rhys Griffiths, Arian Jamasb, Anthony Bourached, Penelope Jones, William McCorkindale, Alexander A. Aldrick, Alpha A. Lee*

1. [**A community-powered search of machine learning strategy space to find NMR property prediction models**](https://arxiv.org/abs/2008.05994), *Lars A. Bratholm, Will Gerrard, Brandon Anderson, Shaojie Bai, Sunghwan Choi, Lam Dang, Pavel Hanchar, Addison Howard, Guillaume Huard, Sanghoon Kim, Zico Kolter, Risi Kondor, Mordechai Kornbluth, Youhan Lee, Youngsoo Lee, Jonathan P. Mailoa, Thanh Tu Nguyen, Milos Popovic, Goran Rakocevic, Walter Reade, Wonho Song, Luka Stojanovic, Erik H. Thiede, Nebojsa Tijanic, Andres Torrubia, Devin Willmott, Craig P. Butts, David R. Glowacki, Kaggle participants*

1. [**Adaptive Layout Decomposition with Graph Embedding Neural Networks**](http://www.cse.cuhk.edu.hk/~byu/papers/C98-DAC2020-MPL-Selector.pdf), *Wei Li, Jialu Xia, Yuzhe Ma, Jialu Li, Yibo Lin, Bei Yu*, DAC'20

1. [**Transfer Learning with Graph Neural Networks for Optoelectronic Properties of Conjugated Oligomers**](https://aip.scitation.org/doi/10.1063/5.0037863), J. Chem. Phys. 154, *Chee-Kong Lee, Chengqiang Lu, Yue Yu, Qiming Sun, Chang-Yu Hsieh, Shengyu Zhang, Qi Liu, and  Liang Shi*

1. [**Jet tagging in the Lund plane with graph networks**](https://link.springer.com/article/10.1007/JHEP03(2021)052), Journal of High Energy Physics 2021, *Frédéric A. Dreyer and Huilin Qu*

1. [**Global Attention Improves Graph Networks Generalization**](https://arxiv.org/abs/2006.07846), *Omri Puny, Heli Ben-Hamu, and Yaron Lipman*

1. [**Learning over Families of Sets -- Hypergraph Representation Learning for Higher Order Tasks**](https://arxiv.org/abs/2101.07773), SDM 2021, *Balasubramaniam Srinivasan, Da Zheng, and George Karypis*

1. [**SSFG: Stochastically Scaling Features and Gradients for Regularizing Graph Convolution Networks**](https://arxiv.org/abs/2102.10338), *Haimin Zhang, Min Xu*

1. [**Application and evaluation of knowledge graph embeddings in biomedical data**](https://peerj.com/articles/cs-341/), PeerJ Computer Science 7:e341, *Mona Alshahrani​, Maha A. Thafar, Magbubah Essack*

1. [**MoTSE: an interpretable task similarity estimator for small molecular property prediction tasks**](https://www.biorxiv.org/content/10.1101/2021.01.13.426608v2), bioRxiv 2021.01.13.426608, *Han Li, Xinyi Zhao, Shuya Li, Fangping Wan, Dan Zhao, Jianyang Zeng*

1. [**Reinforcement Learning For Data Poisoning on Graph Neural Networks**](https://arxiv.org/abs/2102.06800), *Jacob Dineen, A S M Ahsan-Ul Haque, Matthew Bielskas*

1. [**Generalising Recursive Neural Models by Tensor Decomposition**](https://github.com/danielecastellana22/tensor-tree-nn), IJCNN'20, *Daniele Castellana, Davide Bacciu*

1. [**Tensor Decompositions in Recursive Neural Networks for Tree-Structured Data**](https://github.com/danielecastellana22/tensor-tree-nn), ESANN'20, *Daniele Castellana, Davide Bacciu*

1. [**Combining Self-Organizing and Graph Neural Networks for Modeling Deformable Objects in Robotic Manipulation**](https://www.ncbi.nlm.nih.gov/pmc/articles/PMC7806087/), Frotiers in Robotics and AI, *Valencia, Angel J., and Pierre Payeur*

1. [**Joint stroke classification and text line grouping in online handwritten documents with edge pooling attention networks**](https://www.sciencedirect.com/science/article/abs/pii/S0031320321000467), Pattern Recognition, *Jun-Yu Ye, Yan-Ming Zhang, Qing Yang, Cheng-Lin Liu*

1. [**Toward Accurate Predictions of Atomic Properties via Quantum Mechanics Descriptors Augmented Graph Convolutional Neural Network: Application of This Novel Approach in NMR Chemical Shifts Predictions**](https://pubs.acs.org/doi/full/10.1021/acs.jpclett.0c02654), The Journal of Physical Chemistry Letters, *Peng Gao, Jie Zhang, Yuzhu Sun, and Jianguo Yu*

1. [**A Graph Neural Network to Model User Comfort in Robot Navigation**](https://arxiv.org/abs/2102.08863), *Pilar Bachiller, Daniel Rodriguez-Criado, Ronit R. Jorvekar, Pablo Bustos, Diego R. Faria, Luis J. Manso*

1. [**Medical Entity Disambiguation Using Graph Neural Networks**](https://arxiv.org/abs/2104.01488), *Alina Vretinaris, Chuan Lei, Vasilis Efthymiou, Xiao Qin, Fatma Özcan*

1. [**Chemistry-informed Macromolecule Graph Representation for Similarity Computation and Supervised Learning**](https://arxiv.org/abs/2103.02565), *Somesh Mohapatra, Joyce An, Rafael Gómez-Bombarelli*

1. [**Characterizing and Forecasting User Engagement with In-app Action Graph: A Case Study of Snapchat**](https://arxiv.org/pdf/1906.00355.pdf), *Yozen Liu, Xiaolin Shi, Lucas Pierce, Xiang Ren*

1. [**GIPA: General Information Propagation Algorithm for Graph Learning**](https://arxiv.org/abs/2105.06035), *Qinkai Zheng, Houyi Li, Peng Zhang, Zhixiong Yang, Guowei Zhang, Xintan Zeng, Yongchao Liu*

1. [**Graph Ensemble Learning over Multiple Dependency Trees for Aspect-level Sentiment Classification**](https://arxiv.org/abs/2103.11794), NAACL'21, *Xiaochen Hou, Peng Qi, Guangtao Wang, Rex Ying, Jing Huang, Xiaodong He, Bowen Zhou*

1. [**Enhancing Scientific Papers Summarization with Citation Graph**](https://arxiv.org/abs/2104.03057), AAAI'21, *Chenxin An, Ming Zhong, Yiran Chen, Danqing Wang, Xipeng Qiu, Xuanjing Huang*

1. [**Improving Graph Representation Learning by Contrastive Regularization**](https://arxiv.org/pdf/2101.11525.pdf), *Kaili Ma, Haochen Yang, Han Yang, Tatiana Jin, Pengfei Chen, Yongqiang Chen, Barakeel Fanseu Kamhoua, James Cheng*

1. [**Extract the Knowledge of Graph Neural Networks and Go Beyond it: An Effective Knowledge Distillation Framework**](https://arxiv.org/pdf/2103.02885.pdf), WWW'21, *Cheng Yang, Jiawei Liu, Chuan Shi*

1. [**VIKING: Adversarial Attack on Network Embeddings via Supervised Network Poisoning**](https://arxiv.org/pdf/2102.07164.pdf), PAKDD'21, *Viresh Gupta, Tanmoy Chakraborty*

1. [**Knowledge Graph Embedding using Graph Convolutional Networks with Relation-Aware Attention**](https://arxiv.org/pdf/2102.07200.pdf), *Nasrullah Sheikh, Xiao Qin, Berthold Reinwald, Christoph Miksovic, Thomas Gschwind, Paolo Scotton*

1. [**SLAPS: Self-Supervision Improves Structure Learning for Graph Neural Networks**](https://arxiv.org/pdf/2102.05034.pdf), *Bahare Fatemi, Layla El Asri, Seyed Mehran Kazemi*

1. [**Finding Needles in Heterogeneous Haystacks**](https://homepage.divms.uiowa.edu/~badhikari/assets/doc/papers/CONGCNIAAI2021.pdf), AAAI'21, *Bijaya Adhikari, Liangyue Li, Nikhil Rao, Karthik Subbian*

1. [**RetCL: A Selection-based Approach for Retrosynthesis via Contrastive Learning**](https://arxiv.org/abs/2105.00795), IJCAI 2021, *Hankook Lee, Sungsoo Ahn, Seung-Woo Seo, You Young Song, Eunho Yang, Sung-Ju Hwang, Jinwoo Shin*

1. [**Accurate Prediction of Free Solvation Energy of Organic Molecules via Graph Attention Network and Message Passing Neural Network from Pairwise Atomistic Interactions**](https://arxiv.org/abs/2105.02048), *Ramin Ansari, Amirata Ghorbani*

1. [**DIPS-Plus: The Enhanced Database of Interacting Protein Structures for Interface Prediction**](https://arxiv.org/abs/2106.04362), *Alex Morehead, Chen Chen, Ada Sedova, Jianlin Cheng*

1. [**Coreference-Aware Dialogue Summarization**](https://arxiv.org/abs/2106.08556), SIGDIAL'21, *Zhengyuan Liu, Ke Shi, Nancy F. Chen*

1. [**Document Structure aware Relational Graph Convolutional Networks for Ontology Population**](https://arxiv.org/abs/2104.12950), arXiv, *Abhay M Shalghar, Ayush Kumar, Balaji Ganesan, Aswin Kannan, Shobha G*

1. [**Covid-19 Detection from Chest X-ray and Patient Metadata using Graph Convolutional Neural Networks**](https://arxiv.org/abs/2105.09720), *Thosini Bamunu Mudiyanselage, Nipuna Senanayake, Chunyan Ji, Yi Pan, Yanqing Zhang*

1. [**Rossmann-toolbox: a deep learning-based protocol for the prediction and design of cofactor specificity in Rossmann fold proteins**](https://academic.oup.com/bib/advance-article/doi/10.1093/bib/bbab371/6375059), Briefings in Bioinformatics, *Kamil Kaminski, Jan Ludwiczak, Maciej Jasinski, Adriana Bukala, Rafal Madaj, Krzysztof Szczepaniak, Stanislaw Dunin-Horkawicz*

1. [**LGESQL: Line Graph Enhanced Text-to-SQL Model with Mixed Local and Non-Local Relations**](https://arxiv.org/pdf/2106.01093.pdf), ACL'21, *Ruisheng Cao, Lu Chen, Zhi Chen, Yanbin Zhao, Su Zhu, Kai Yu*

1. [**Enhancing Graph Neural Networks via auxiliary training for semi-supervised node classification**](https://www.sciencedirect.com/science/article/pii/S0950705121001477), Knowledge-Based System'21, *Yao Wu, Yu Song, Hong Huang, Fanghua Ye, Xing Xie, Hai Jin*

1. [**Modeling Graph Node Correlations with Neighbor Mixture Models**](https://arxiv.org/pdf/2103.15966.pdf), *Linfeng Liu, Michael C. Hughes, Li-Ping Liu*

1. [**COMBINING PHYSICS AND MACHINE LEARNING FOR NETWORK FLOW ESTIMATION**](https://openreview.net/pdf/9dc2744a465941220de07cf308acf822ec8aaa64.pdf), ICLR'21, *Arlei Silva, Furkan Kocayusufoglu, Saber Jafarpour, Francesco Bullo, Ananthram Swami, Ambuj Singh*

1. [**A Classification Method for Academic Resources Based on a Graph Attention Network**](https://www.mdpi.com/1999-5903/13/3/64/htm), Future Internet'21, *Jie Yu, Yaliu Li, Chenle Pan and Junwei Wang*

1. [**Large Graph Convolutional Network Training with GPU-Oriented Data Communication Architecture**](https://arxiv.org/abs/2103.03330), *Seung Won Min, Kun Wu, Sitao Huang, Mert Hidayetoğlu, Jinjun Xiong, Eiman Ebrahimi, Deming Chen, Wen-mei Hwu*

1. [**Graph Attention Multi-Layer Perception**](https://github.com/PKU-DAIR/GAMLP/blob/main/GAMLP.pdf), *Wentao Zhang, Ziqi Yin, Zeang Sheng, Wen Ouyang, Xiaosen Li, Yangyu Tao, Zhi Yang, Bin Cui*

1. [**GNNLens: A Visual Analytics Approach for Prediction Error Diagnosis of Graph Neural Networks**](https://arxiv.org/abs/2011.11048v5), *Zhihua Jin, Yong Wang, Qianwen Wang, Yao Ming, Tengfei Ma, Huamin Qu*

1. [**How Attentive are Graph Attention Networks?**](https://arxiv.org/pdf/2105.14491.pdf), *Shaked Brody, Uri Alon, Eran Yahav*, [code](https://github.com/tech-srl/how_attentive_are_gats)

1. [**SCENE: Reasoning about Traffic Scenes using Heterogeneous Graph Neural Networks**](https://arxiv.org/pdf/2301.03512.pdf), *Thomas Monninger\*, Julian Schmidt\*, Jan Rupprecht, David Raba, Julian Jordan, Daniel Frank, Steffen Staab, Klaus Dietmayer*, [code](https://github.com/schmidt-ju/scene), \*co-first authors

</details>

## Contributing

Please let us know if you encounter a bug or have any suggestions by [filing an issue](https://github.com/dmlc/dgl/issues).

We welcome all contributions from bug fixes to new features and extensions.

We expect all contributions discussed in the issue tracker and going through PRs.  Please refer to our [contribution guide](https://docs.dgl.ai/contribute.html).

## Cite

If you use DGL in a scientific publication, we would appreciate citations to the following paper:
```
@article{wang2019dgl,
    title={Deep Graph Library: A Graph-Centric, Highly-Performant Package for Graph Neural Networks},
    author={Minjie Wang and Da Zheng and Zihao Ye and Quan Gan and Mufei Li and Xiang Song and Jinjing Zhou and Chao Ma and Lingfan Yu and Yu Gai and Tianjun Xiao and Tong He and George Karypis and Jinyang Li and Zheng Zhang},
    year={2019},
    journal={arXiv preprint arXiv:1909.01315}
}
```

## The Team

DGL is developed and maintained by [NYU, NYU Shanghai, AWS Shanghai AI Lab, and AWS MXNet Science Team](https://www.dgl.ai/pages/about.html).

## License

DGL uses Apache License 2.0.
