Inductive Representation Learning on Large Graphs (GraphSAGE)
============

- Paper link: [http://papers.nips.cc/paper/6703-inductive-representation-learning-on-large-graphs.pdf](http://papers.nips.cc/paper/6703-inductive-representation-learning-on-large-graphs.pdf)
- Author's code repo: [https://github.com/williamleif/graphsage-simple](https://github.com/williamleif/graphsage-simple). Note that the original code is 
simple reference implementation of GraphSAGE.

Requirements
------------
- requests

``bash
pip install requests
``


Results
-------

Run with following (available dataset: "cora", "citeseer", "pubmed")
```bash
python3 graphsage.py --dataset cora --gpu 0
```

* cora: ~0.8330 
* citeseer: ~0.7110
* pubmed: ~0.7830

### Minibatch training with neighbor sampling on Reddit

Run with following for Reddit with minibatch training:
```bash
python3 graphsage_sampling.py
```

Run with following for Reddit with minibatch training on 4 GPUs:
```bash
python3 graphsage_sampling.py --gpu 0,1,2,3
```

The accuracy can go to around 0.95.
