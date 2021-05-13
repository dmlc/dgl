# Instructions to download datasets:

1. Download datasets from here: https://www.dropbox.com/s/verx1evkykzli88/datasets.zip
2. Extract zip folder in this directory
3. Choose the dataset you wish in `run.py` file. 

# Details about BGNN model
`run.py` implements a class for GNN model. You can select GAT, GCN, ChebNet, AGNN, or APPNP gnn models.
Or you can provide your favorite GNN model. You can also pretrain your model or setup the hyperparameters you like. 

Hyperparameters of BGNN model. 
* `append_gbdt_pred` -- this decides whether to append GBDT predictions from GNN to original input features or to replace original input features with predictions of GBDT. This can be important for performance, so try both values, True and False. 
* `trees_per_epoch` and `backprop_per_epoch`. Values in the range 5-15 usually gives good results. The more, the longer training is. 
* `lr` is learning rate for GNN. 0.01-0.1 are good values to try.
* `gbdt_lr` is learning rate for GBDT. Should be that important. 
* `gbdt_depth` number of levels in GBDT tree. 4-8 are good values. The more, the longer it trains. 