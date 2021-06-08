# DGL Implementation of Graph Cross Networks with Vertex Infomax Pooling (NeurIPS 2020)

This DGL example implements the GNN model proposed in the paper [Graph Cross Networks with Vertex Infomax Pooling](https://arxiv.org/pdf/2010.01804.pdf). 
The author's codes of implementation is in [here](https://github.com/limaosen0/GXN)


The graph dataset used in this example 
---------------------------------------
The DGL's built-in LegacyTUDataset. This is a serial of graph kernel datasets for graph classification. We use 'DD', 'PROTEINS', 'ENZYMES', 'IMDB-BINARY', 'IMDB-MULTI' and 'COLLAB' in this GXN implementation. All these datasets are randomly splited to train and test set with ratio 0.9 and 0.1 (which is similar to the setting in the author's implementation).

NOTE: Follow the setting of the author's implementation, for 'DD' and 'PROTEINS', we use one-hot node label as input node features. For ENZYMES', 'IMDB-BINARY', 'IMDB-MULTI' and 'COLLAB', we use the concatenation of one-hot node label (if available) and one-hot node degree as input node features.

|                  | DD     | PROTEINS | ENZYMES | IMDB-BINARY  | IMDB-MULTI | COLLAB   |
| ---------------- | ------ | -------- | ------- | ------------ | ---------- | -------- |
| NumGraphs        | 1178   | 1113     | 600     | 1000         | 1500       | 5000     |
| AvgNodesPerGraph | 284.32 | 39.06    | 32.63   | 19.77        | 13.00      | 74.49    |
| AvgEdgesPerGraph | 715.66 | 72.82    | 62.14   | 96.53        | 65.94      | 2457.78  |
| NumFeats         | 89     | 1        | 18      | -            | -          | -        |
| NumClasses       | 2      | 2        | 6       | 2            | 3          | 2        |


How to run example files
--------------------------------
If you want to reproduce the author's result, at the root directory of this example (gxn), run

```bash
bash scripts/run_gxn.sh ${dataset_name} ${device_id} ${num_trials} ${print_trainlog_every}
```

If you want to perform a early-stop version experiment, at the root directory of this example, run

```bash
bash scripts/run_gxn_early_stop.sh ${dataset_name} ${device_id} ${num_trials} ${print_trainlog_every}
```

where
- dataset_name: Dataset name used in this experiment. Could be DD', 'PROTEINS', 'ENZYMES', 'IMDB-BINARY', 'IMDB-MULTI' and 'COLLAB'.
- device_id: ID of computation device. -1 for pure CPU computation. For example if you only have single GPU, set this value to be 0.
- num_trials: How many times does the experiment conducted.
- print_training_log_every: Print training log every ? epochs. -1 for silent training.


NOTE: If your have problem when using 'IMDB-BINARY', 'IMDB-MULTI' and 'COLLAB', it could be caused by a bug in `LegacyTUDataset`/`TUDataset` in DGL (see [here](https://github.com/dmlc/dgl/pull/2543)). If your DGL version is less than or equal to 0.5.3 and you encounter problems like "undefined variable" (`LegacyTUDataset`) or "the argument `force_reload=False` does not work" (`TUDataset`), try:
- use `TUDataset` with `force_reload=True`
- delete dataset files 
- change `degree_as_feature(dataset)` and `node_label_as_feature(dataset, mode=mode)` to `degree_as_feature(dataset, save=False)` and `node_label_as_feature(dataset, mode=mode, save=False)` in `main.py`.

Performance
-------------------------

**Accuracy**

**NOTE**: Different from our implementation, the author uses fixed dataset split. Thus there may be difference between our result and the author's result. **To compare our implementation with the author's, we follow the setting in the author's implementation that performs model-selection on testset**. We also try early-stop with patience equals to 1/5 of the total number of epochs for some datasets. The result of `Author's Code` in the table below are obtained using first-ford data as the test dataset.

|                   | DD           | PROTEINS    | ENZYMES     | IMDB-BINARY | IMDB-MULTI | COLLAB     |
| ------------------| ------------ | ----------- | ----------- | ----------- | ---------- | ---------- |
| Reported in Paper | 82.68(4.1 )  | 79.91(4.1)  | 57.50(6.1)  | 78.60(2.3)  | 55.20(2.5) | 78.82(1.4) |
| Author's Code     | 82.05        | 72.07       | 58.33       | 77.00       | 56.00      | 80.40      |
| DGL               | 82.97(3.0)   | 78.21(2.0)  | 57.50(5.5)  | 78.70(4.0)  | 52.26(2.0) | 80.58(2.4) |
| DGL(early-stop)   | 78.66(4.3)   | 73.12(3.1)  | 39.83(7.4)  | 68.60(6.7)  | 45.40(9.4) | 76.18(1.9) |


**Speed**

Device: 
- CPU: Intel(R) Xeon(R) CPU E5-2686 v4 @ 2.30GHz
- GPU: Tesla V100-SXM2 16GB

In seconds

|               | DD    | PROTEINS | ENZYMES | IMDB-BINARY | IMDB-MULTI | COLLAB(batch_size=64) | COLLAB(batch_size=20) |
| ------------- | ----- | -------- | ------- | ----------- | ---------- | --------------------- | --------------------- |
| Author's Code | 25.32 | 2.93     | 1.53    | 2.42        | 3.58       | 96.69                 | 19.78                 |
| DGL           | 2.64  | 1.86     | 1.03    | 1.79        | 2.45       | 23.52                 | 32.29                 |
