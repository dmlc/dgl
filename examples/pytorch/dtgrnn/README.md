# Discrete Temporal Dynamic Graph with recurrent structure
## DGL Implementation of DCRNN and GaAN paper.

This DGL example implements the GNN model proposed in the paper [Diffusion Convolutional Recurrent Neural Network: Data-Driven Traffic Forecasting](https://arxiv.org/abs/1707.01926) and [GaAN:Gated Attention Networks for Learning on Large and Spatiotemporal Graphs](https://arxiv.org/pdf/1803.07294). 

Model implementor
----------------------
This example was implemented by [Ericcsr](https://github.com/Ericcsr) during his Internship work at the AWS Shanghai AI Lab.

The graph dataset used in this example 
---------------------------------------
METR-LA dataset. Dataset summary:
- NumNodes: 207
- NumEdges: 1722
- NumFeats: 2
- TrainingSamples: 70%
- ValidationSamples: 20%
- TestSamples: 10%

PEMS-BAY dataset. Dataset Summary:

- NumNodes: 325
- NumEdges: 2694
- NumFeats: 2
- TrainingSamples: 70%
- ValidationSamples: 20%
- TestSamples: 10%

How to run example files
--------------------------------
In the dtdg folder, run

**Please use `train.py`**

Train the DCRNN model on METR-LA Dataset

```python
python train.py --dataset LA --model dcrnn
```

If want to use a GPU, run

```python
python train.py --gpu 0 --dataset LA --model dcrnn
```

if you want to use PEMS-BAY dataset

```python
python train.py --gpu 0 --dataset BAY --model dcrnn
```

Train GaAN model

```python
python train.py --gpu 0 --model gaan --dataset <LA/BAY>
```


Performance on METR-LA
-------------------------
| Models/Datasets | Test MAE |
| :-------------- | --------:|
| DCRNN in DGL    | 2.91 |
| DCRNN paper     | 3.17 |
| GaAN in DGL     | 3.20 |
| GaAN paper      | 3.16 |


Notice that Any Graph Convolution module can be plugged into the recurrent discrete temporal dynamic graph template to test performance; simply replace DiffConv or GaAN.

