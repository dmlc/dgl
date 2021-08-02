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
- dgl 0.7.0
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
To run the full graph version, in the care-gnn folder, run
```
python main.py
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
The result reported by the paper is the best validation results within 30 epochs, while ours are testing results after the max epoch specified in the table. Early stopping with patience value of 100 is applied.

<table>
	<tr>
	    <th colspan="2">Dataset</th>
	    <th>Amazon</th>
	    <th>Yelp</th>
	</tr >
	<tr>
        <td>Metric</td>
        <td>Max Epoch</td>
	    <td>30 / 1000</td>
	    <td>30 / 1000</td>
	</tr>
	<tr >
	    <td rowspan="3">AUC</td>
	    <td>paper reported</td>
	    <td>89.73 / -</td>
        <td>75.70 / -</td>
	</tr>
	<tr>
	    <td>DGL full graph</td>
	    <td>89.50 / 92.35</td>
	    <td>69.16 / 79.91</td>
	</tr>
	<tr>
	    <td>DGL sampling</td>
	    <td>93.27 / 92.94</td>
        <td>79.38 / 80.53</td>
	</tr>
	<tr >
	    <td rowspan="3">Recall</td>
	    <td>paper reported</td>
	    <td>88.48 / -</td>
        <td>71.92 / -</td>
	</tr>
	<tr>
	    <td>DGL full graph</td>
	    <td>85.54 / 84.47</td>
	    <td>69.91 / 73.47</td>
	</tr>
	<tr>
	    <td>DGL sampling</td>
	    <td>85.83 / 87.46</td>
        <td>77.26 / 64.34</td>
	</tr>
</table>
