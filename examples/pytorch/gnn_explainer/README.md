# DGL Implementation of the GNN Explainer

This DGL example implements the GNN Explainer model proposed in the paper [GNNExplainer: Generating Explanations for Graph Neural Networks](https://arxiv.org/abs/1903.03894). 
The author's codes of implementation is in [here](https://github.com/RexYing/gnn-model-explainer).

The author's implementation is kind of experimental with experimental codes. So this implementation focuses on a subset of
GNN Explainer's functions, node classification, and later on extend to edge classification.

Example implementor
----------------------
This example was implemented by [Jian Zhang](https://github.com/zhjwy9343) and [Kounianhua Du](https://github.com/KounianhuaDu) at the AWS Shanghai AI Lab.

Dependencies
----------------------
- numpy 1.19.4
- pytorch 1.7.1
- dgl 0.5.3
- networkx 2.5
- matplotlib 3.3.4

Datasets
----------------------
Five synthetic datasets used in the paper are used in this example. The generation codes are referenced from the author implementation.
- Syn1 (BA-SHAPES): Start with a base Barabasi-Albert (BA) graph on 300 nodes and a set of 80 five-node “house”-structured network motifs, which are attached to randomly selected nodes of the base graph. The resulting graph is further perturbed by adding 0.01N random edges.  Nodes are assigned to 4 classes based on their structural roles. In a house-structured motif, there are 3 types of roles: the top, middle, and bottom node of the house. Therefore there are 4 different classes, corresponding to nodes at the top, middle, bottom of houses, and nodes that do not belong to a house. 
- Syn2 (BA-COMMUNITY): A union of two BA-SHAPES graphs. Nodes have normally distributed feature vectors and are assigned to one of 8 classes based on their structural roles and community memberships. 
- Syn3 (BA-GRID): The same as BA-SHAPES except that 3-by-3 grid motifs are attached to the base graph in place of house motifs.
- Syn4 (TREE-CYCLE): Start with a base 8-level balanced binary tree and 60 six-node cycle motifs, which are attached to random nodes of the base graph. Perturbed by adding 0.01N random edges.
- Syn5 (TREE-GRID): Start with a base 8-level balanced binary tree and 80 3-by-3 grid motifs, which are attached to random nodes of the base graph. Perturbed by adding 0.1N random edges.

Demo Usage
----------------------
**First**, train a demo GNN model by using a synthetic dataset.
``` python
python train_main.py  --dataset syn1
```
Replace the argument of the --dataset, available options: syn1, syn2, syn3, syn4, syn5

This command trains a GNN model and save it to the "dummy_model_syn1.pth" file.

**Second**, explain the trained model with the same data
``` python
python explain_main.py --dataset syn1 --target_class 1 --hop 2
```
Replace the dataset argument value and the target class you want to explain. The code will pick the first node in the specified class to explain. The --hop argument corresponds to the maximum hop number of the computation sub-graph. (For syn1 and syn2, hop=2. For syn3, syn4, and syn5, hop=4.)

Notice
----------------------
Because DGL does not support masked adjacency matrix as an input to the forward function of a module.
To use this Explainer, you need to add an edge_weight as the **edge mask** argument to your forward function just like 
the dummy model in the models.py file. And you need to change your forward function whenever uses `.update_all` function. 
Please use `dgl.function.u_mul_e` to compute the src nodes' features to the edge_weights as the mask method proposed by the 
GNN Explainer paper. Check the models.py for details.

Results
----------------------
For all the datasets, the first node of target class 1 is picked to be explained. The hop-k computation sub-graph (a compact of 0-hop, 1-hop, ..., k-hop subgraphs) is first extracted and then fed to the models. Followings are the visualization results. Instead of cutting edges that are below the threshold. We use the depth of color of the edges to represent the edge mask weights. The deeper the color of an edge is, the more important the edge is. 

NOTE: We do not perform grid search or finetune here, the visualization results are just for reference.


**Syn1 (BA-SHAPES)**
<p align="center">
  <img src="https://github.com/KounianhuaDu/gnn-explainer-dgl-pics/blob/master/imgs/syn1.png"  width="600">
  <br>
  <b>Figure</b>: Visualization of syn1 dataset (hop=2).
</p>

**Syn2 (BA-COMMUNITY)**
<p align="center">
  <img src="https://github.com/KounianhuaDu/gnn-explainer-dgl-pics/blob/master/imgs/syn2.png"  width="600">
  <br>
  <b>Figure</b>: Visualization of syn2 dataset (hop=2).
</p>

**Syn3 (BA-GRID)**

For a more explict view, we conduct explaination on both the hop-3 computation sub-graph and the hop-4 computation sub-graph in Syn3 task.

<p align="center">
  <img src="https://github.com/KounianhuaDu/gnn-explainer-dgl-pics/blob/master/imgs/syn3_3hop.png"  width="600">
  <br>
  <b>Figure</b>: Visualization of syn3 dataset with hop=3.
</p>

<p align="center">
  <img src="https://github.com/KounianhuaDu/gnn-explainer-dgl-pics/blob/master/imgs/syn3_4hop.png"  width="600">
  <br>
  <b>Figure</b>: Visualization of syn3 dataset with hop=4.
</p>

**Syn4 (TREE-CYCLE)**
<p align="center">
  <img src="https://github.com/KounianhuaDu/gnn-explainer-dgl-pics/blob/master/imgs/syn4.png"  width="600">
  <br>
  <b>Figure</b>: Visualization of syn4 dataset (hop=4).
</p>

**Syn5 (TREE-GRID)**
<p align="center">
  <img src="https://github.com/KounianhuaDu/gnn-explainer-dgl-pics/blob/master/imgs/syn5.png"  width="600">
  <br>
  <b>Figure</b>: Visualization of syn5 dataset (hop=4).
</p>
