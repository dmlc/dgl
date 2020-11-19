# GNN Explainer

In the context of DGL, implement the paper [GNNExplainer: Generating Explanations for Graph Neural Networks](https://arxiv.org/abs/1903.03894).

Author's [codes](https://github.com/RexYing/gnn-model-explainer).

The author's implementation is kind of experimental with unclean codes. So this implementation focuses on a subset of
GNN Explainer's functions, node classification, and later on extend to edge classification.

## Demo Usage

**First**, train a demo GNN model by using a synthetic dataset.
``` python
python train_main.py  --dataset syn1
```
Replace the argument of the --dataset, available options: syn1, syn2, syn3, syn4, syn5

This command trains a GNN model and save it to the "dummy_model_4_syn1.pth" file.

**Second**, explain the trained model with the same data
``` python
python explain_main.py --model_path dummy_model_4_syn1.pth --target_class 1
```
Replace the model_path argument value to the model you saved in the above training process, and the target class you want
to explain. The code will pick the first node in the specified class to explain.

### Notice
Because DGL does not support masked adjacency matrix as an input for the forward function of a module.
To use this Explainer, you need to add an edge_weight as the **edge mask** argument to your forward function just like 
the dummy model in the models.py file. And you need to change your forward function whenever uses `.update_all` function. 
Please use `dgl.function.u_mul_e` to compute the src nodes' features to the edge_weights as the mask method proposed by the 
GNN Explainer paper. Check the models.py for details.