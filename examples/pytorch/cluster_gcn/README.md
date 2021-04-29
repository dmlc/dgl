Cluster-GCN: An Efficient Algorithm for Training Deep and Large Graph Convolutional Networks
============
- Paper link: [Cluster-GCN: An Efficient Algorithm for Training Deep and Large Graph Convolutional Networks](https://arxiv.org/abs/1905.07953)
- Author's code repo: [https://github.com/google-research/google-research/blob/master/cluster_gcn/](https://github.com/google-research/google-research/blob/master/cluster_gcn/). 

This repo reproduce the reported speed and performance maximally on Reddit and PPI. However, the diag enhancement is not covered, as the GraphSage aggregator already achieves satisfying F1 score.

Dependencies
------------
- Python 3.7+(for string formatting features)
- PyTorch 1.5.0+
- sklearn


## Run Experiments.
* For reddit data, you may run the following scripts

```
./run_reddit.sh
```
You should be able to see the final test F1 is around `Test F1-mic0.9612, Test F1-mac0.9399`.
Note that the first run of provided script is considerably slow than reported in the paper, which is presumably due to dataloader used. After caching the partition allocation, the overall speed would be in a normal scale. On a 1080Ti and Intel(R) Xeon(R) Bronze 3104 CPU @ 1.70GHz machine I am able to train it within 45s. After the first epoch the F1-mic on Validation dataset should be around `0.93`.

* For PPI data, you may run the following scripts

```
./run_ppi.sh
```
You should be able to see the final test F1 is around `Test F1-mic0.9924, Test F1-mac0.9917`. The training finished in 10 mins.
