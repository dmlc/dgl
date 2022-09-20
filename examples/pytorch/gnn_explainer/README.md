# DGL Implementation of GNNExplainer

This is a DGL example for [GNNExplainer: Generating Explanations for Graph Neural Networks](https://arxiv.org/abs/1903.03894). For the authors' original implementation,
see [here](https://github.com/RexYing/gnn-model-explainer).

Contributors:
- [Jian Zhang](https://github.com/zhjwy9343)
- [Kounianhua Du](https://github.com/KounianhuaDu)
- [Yanjun Zhao](https://github.com/zyj-111)

Datasets
----------------------

Four built-in synthetic datasets are used in this example.

- [BA-SHAPES](https://docs.dgl.ai/generated/dgl.data.BAShapeDataset.html#dgl.data.BAShapeDataset)
- [BA-COMMUNITY](https://docs.dgl.ai/generated/dgl.data.BACommunityDataset.html#dgl.data.BACommunityDataset)
- [TREE-CYCLE](https://docs.dgl.ai/generated/dgl.data.TreeCycleDataset.html#dgl.data.TreeCycleDataset)
- [TREE-GRID](https://docs.dgl.ai/generated/dgl.data.TreeGridDataset.html#dgl.data.TreeGridDataset)

Usage
----------------------

**First**, train a GNN model on a dataset.

```bash
python train_main.py  --dataset $DATASET
```

Valid options for `$DATASET`: `BAShape`, `BACommunity`, `TreeCycle`, `TreeGrid`

The trained model weights will be saved to `model_{dataset}.pth`

**Second**, install [GNNLens2](https://github.com/dmlc/GNNLens2) with

```bash
pip install -U flask-cors
pip install Flask==2.0.3
pip install gnnlens
```

**Third**, explain the trained model with the same dataset

```bash
python explain_main.py --dataset $DATASET
```

**Finally**, launch `GNNLens2` to visualize the explanations

```bash
gnnlens --logdir gnn_subgraph
```

By entering `localhost:7777` in your web browser address bar, you can see the GNNLens2 interface. `7777` is the default port GNNLens2 uses. You can specify an alternative one by adding `--port xxxx` after the command line and change the address in the web browser accordingly.

A sample visualization is available below. For more details of using `GNNLens2`, check its [tutorials](https://github.com/dmlc/GNNLens2#tutorials).

<p align="center">
  <img src="https://data.dgl.ai/asset/image/explain_BAShape.png"  width="600">
  <br>
  <b>Figure</b>: Explanation for node 41 of BAShape
</p>
