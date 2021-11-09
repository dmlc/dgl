# DGL Implementation of the CARE-GNN Paper

This DGL example implements the CAmouflage-REsistant GNN (CARE-GNN) model proposed in the paper [Enhancing Graph Neural Network-based Fraud Detectors against Camouflaged Fraudsters](https://arxiv.org/abs/2008.08692). The author's codes of implementation is [here](https://github.com/YingtongDou/CARE-GNN).

**NOTE**: The sampling version of this model has been modified according to the feature of the DGL's NodeDataLoader. For the formula 2 in the paper, rather than using the embedding of the last layer, this version uses the embedding of the current layer in the previous epoch to measure the similarity between center nodes and their neighbors.

Example implementor
----------------------
This example was implemented by [Kay Liu](https://github.com/kayzliu) during his SDE intern work at the AWS Shanghai AI Lab.

Dependencies
----------------------
- Python 3.7.10
- PyTorch 1.8.1
- dgl 0.7.1
- scikit-learn 0.23.2

Dataset
---------------------------------------
The datasets used for node classification are DGL's built-in FraudDataset. The statistics are summarized as followings:

**Amazon**

- Nodes: 11,944
- Edges:
    - U-P-U: 351,216
    - U-S-U: 7,132,958
    - U-V-U: 2,073,474
- Classes:
    - Positive (fraudulent): 821
    - Negative (benign): 7,818
    - Unlabeled: 3,305
- Positive-Negative ratio: 1 : 10.5
- Node feature size: 25

**YelpChi**

- Nodes: 45,954
- Edges:
    - R-U-R: 98,630
    - R-T-R: 1,147,232
    - R-S-R: 6,805,486
- Classes:
    - Positive (spam): 6,677
    - Negative (legitimate): 39,277
- Positive-Negative ratio: 1 : 5.9
- Node feature size: 32

How to run
--------------------------------
To run the full graph version and use early stopping, in the care-gnn folder, run
```
python main.py --early-stop
```

If want to use a GPU, run
```
python main.py --gpu 0
```

To train on Yelp dataset instead of Amazon, run
```
python main.py --dataset yelp
```

To run the sampling version, run
```
python main_sampling.py
```

Performance
-------------------------
The result reported by the paper is the best validation results within 30 epochs, and the table below reports the val and test results (same setting in the paper except for the random seed, here `seed=717`). 

<table>
<thead>
  <tr>
    <th colspan="2">Dataset</th>
    <th>Amazon</th>
    <th>Yelp</th>
  </tr>
</thead>
<tbody>
  <tr>
    <td>Metric (val / test)</td>
    <td>Max Epoch</td>
    <td>30</td>
    <td>30 </td>
  </tr>
  <tr>
    <td rowspan="3">AUC (val/test)</td>
    <td>paper reported</td>
    <td>0.8973 / -</td>
    <td>0.7570 / -</td>
  </tr>
  <tr>
    <td>DGL full graph</td>
    <td>0.8849 / 0.8922</td>
    <td>0.6856 / 0.6867</td>
  </tr>
  <tr>
    <td>DGL sampling</td>
    <td>0.9350 / 0.9331</td>
    <td>0.7857 / 0.7890</td>
  </tr>
  <tr>
    <td rowspan="3">Recall (val/test)</td>
    <td>paper reported</td>
    <td>0.8848 / -</td>
    <td>0.7192 / -</td>
  </tr>
  <tr>
    <td>DGL full graph</td>
    <td>0.8615 / 0.8544</td>
    <td>0.6667/ 0.6619</td>
  </tr>
  <tr>
    <td>DGL sampling</td>
    <td>0.9130 / 0.9045</td>
    <td>0.7537 / 0.7540</td>
  </tr>
</tbody>
</table>



