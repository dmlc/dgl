# DGL Implementation of the SEAL Paper
This DGL example implements the link prediction model proposed in the paper 
[Link Prediction Based on Graph Neural Networks](https://arxiv.org/pdf/1802.09691.pdf) 
and [REVISITING GRAPH NEURAL NETWORKS FOR LINK PREDICTION](https://arxiv.org/pdf/2010.16103.pdf)  
The author's codes of implementation is in [SEAL](https://github.com/muhanzhang/SEAL) (pytorch)
and [SEAL_ogb](https://github.com/facebookresearch/SEAL_OGB) (torch_geometric)

Example implementor
----------------------
This example was implemented by [Smile](https://github.com/Smilexuhc) during his intern work at the AWS Shanghai AI Lab.

The graph dataset used in this example 
---------------------------------------

ogbl-collab
 - NumNodes: 235868
 - NumEdges: 2358104
 - NumNodeFeats: 128
 - NumEdgeWeights: 1
 - NumValidEdges: 160084
 - NumTestEdges: 146329

Dependencies
--------------------------------

- python 3.6+
- Pytorch 1.5.0+
- dgl 0.6.0 +
- ogb  
- pandas
- tqdm
- scipy


 How to run example files
--------------------------------
In the seal_dgl folder    
run on cpu:  
```shell script
python main.py --gpu_id=-1 --subsample_ratio=0.1
```
run on gpu:  
```shell script
python main.py --gpu_id=0  --subsample_ratio=0.1
```

Performance
-------------------------
experiment on `ogbl-collab`

| method | valid-hits@50 | test-hits@50 |
| ------ | ------------- | ------------ |
| paper  | 63.89(0.49)         | 53.71(0.47)        |
| ours     | 63.56(0.71)         | 53.61(0.78)        |

Note: We only perform 5 trails in the experiment. 