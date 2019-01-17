# Model Examples using DGL (w/ MXNet backend)

use `DGLBACKEND=mxnet` to use MXNet as DGL's backend

## Examples:

```
DGLBACKEND=mxnet python gcn_batch.py --dataset cora
DGLBACKEND=mxnet python gat_batch.py --dataset cora
```

Each model is hosted in their own folders. Please read their README.md to see how to
run them.

To understand step-by-step how these models are implemented in DGL. Check out our
[tutorials](https://docs.dgl.ai/tutorials/models/index.html)
