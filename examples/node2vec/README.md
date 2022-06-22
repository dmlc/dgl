# DGL Implementation of the Node2vec
This DGL example implements the graph embedding model proposed in the paper 
[node2vec: Scalable Feature Learning for Networks](https://arxiv.org/abs/1607.00653) 

The author's codes of implementation is in [Node2vec](https://github.com/aditya-grover/node2vec) 


Example implementor
----------------------
This example was implemented by [Smile](https://github.com/Smilexuhc) during his intern work at the AWS Shanghai AI Lab.

The graph dataset used in this example 
---------------------------------------

cora
 - NumNodes: 2708
 - NumEdges: 10556

ogbn-products
 - NumNodes: 2449029
 - NumEdges: 61859140

 
Dependencies
--------------------------------

- python 3.6+
- Pytorch 1.5.0+
- ogb  


 How to run example files
--------------------------------
To train a node2vec model:
```shell script
python main.py --task="train"
```

To time node2vec random walks:
```shell script
python main.py --task="time" --runs=10
```

Performance
-------------------------

**Setting:** `walk_length=50, p=0.25, q=4.0`
| Dataset  |     DGL     |     PyG     |
| -------- | :---------: | :---------: |
| cora     | 0.0092s | 0.0179s |
| products | 66.22s  | 77.65s  |
Note that the number in table are the average results of multiple trials.  
For cora, we run 50 trials.  For ogbn-products, we run 10 trials.
