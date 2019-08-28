
# Customize Dataset

Generally we follow the practise of PyTorch. 

A Dataset class should implement `__getitem__(self, index)` and `__len__(self)`method

```python
class CustomDataset:
    def __init__(self):
        # Initialize Dataset and preprocess data

    def __getitem__(self, index):
        # Return the corresponding DGLGraph/label needed for training/evaluation based on index
        return self.graphs[index], self.labels[index]

    def __len__(self):
        return len(self.graphs)
```

DGL supports various backends such as MXNet and PyTorch, therefore we want our dataset to be also backend agnostic.
We prefer user using numpy array in the dataset, and not including any operator/tensor from the specific backend. 
If you want to convert the numpy array to the corresponding tensor, you can use the following code

```python
import dgl.backend as F

# g is a DGLGraph, h is a numpy array
g.ndata['h'] = F.zerocopy_from_numpy(h)
# Now g.ndata is a PyTorch Tensor or a MXNet NDArray based on backend used 
```

If your dataset is in `.csv` format, you may use
[`CSVDataset`](https://github.com/dmlc/dgl/blob/master/python/dgl/data/chem/csv_dataset.py).
