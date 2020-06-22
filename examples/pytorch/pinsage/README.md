# PinSAGE example

## Prepare datasets

### MovieLens 1M

1. Download and extract the MovieLens-1M dataset from http://files.grouplens.org/datasets/movielens/ml-1m.zip
   into the current directory.
2. Run `python process_movielens1m.py ./ml-1m ./data.pkl`.
   Replace `ml-1m` with the directory you put the `.dat` files, and replace `data.pkl` to
   any path you wish to put the output pickle file.

### Nowplaying-rs

1. Download and extract the Nowplaying-rs dataset from https://zenodo.org/record/3248543/files/nowplayingrs.zip?download=1
   into the current directory.
2. Run `python preprocess_nowplaying_rs.py ./nowplaying_rs_dataset ./data.pkl 

## Run model

### Nearest-neighbor recommendation

This model returns items that are K nearest neighbors of the latest item the user has
interacted.  The distance between two items are measured by Euclidean distance of
item embeddings, which are learned as outputs of PinSAGE.

```
python model.py data.pkl --num-epochs 300 --num-workers 2 --device cuda:0 --hidden-dims 64
```

The implementation here also assigns a learnable vector to each item.  If your hidden
state size is so large that the learnable vectors cannot fit into GPU, use this script
for sparse embedding update (written with `torch.optim.SparseAdam`) instead:


```
python model_sparse.py data.pkl --num-epochs 300 --num-workers 2 --device cuda:0 --hidden-dims 1024
```

Note that since the embedding update is done on CPU, it will be significantly slower than doing
everything on GPU.

The HITS@10 is 0.01241, compared to 0.01220 with SLIM with the same dimensionality.
