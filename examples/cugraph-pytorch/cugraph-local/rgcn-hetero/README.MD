# Relational-GCN on cugraph_storage
This readme just extends: https://github.com/dmlc/dgl/blob/master/examples/pytorch/rgcn-hetero/README.md
to work with cugraphstorage 


### Dependencies
* PyTorch 1.0+
* requests
* rdflib
* cugraph 22.12.00a221022+

```
mamba create -n dgl_and_cugraph -c dglteam -c rapidsai-nightly -c nvidia -c pytorch -c conda-forge     cugraph pytorch cudatoolkit=11.6 dgl-cuda11.6 tqdm rdflib
```

### Results:

| dataset | Avg(cugraph, 18 runs) | Avg(DGL, 5 runs) |
|---------|-----------------------|------------------|
| aifb    | 93.81                 | 94.44            |
| bgs     | 90.04                 | 91.04            |
| mutag   | 66.17                 | 67.37            |


### Entity Classification w/ minibatch training with cugraph

Just add ```--use_cugraph_storage``` which converts ```dgl.heterograph``` to ```dgl.contrib.cugraph.cugraph_storage.CuGraphStorage``` . 
```python
from dgl.contrib.cugraph.convert import cugraph_storage_from_heterograph
g = cugraph_storage_from_heterograph(g)
```

Accuracy numbers are reported by 5 runs.

AIFB:
```
python3 entity_classify_mb.py -d aifb --testing --gpu 0 --fanout=8 --use_cugraph_storage
```
MUTAG:
```
python3 entity_classify_mb.py -d mutag --l2norm 5e-4 --n-bases 30 --testing --gpu 0 --batch-size=50 --fanout=8 --use_cugraph_storage
```
BGS:
```
python3 entity_classify_mb.py -d bgs --l2norm 5e-4 --n-bases 40 --testing --gpu 0 --use_cugraph_storage
```

AM: 
```
python3 entity_classify_mb.py -d am --l2norm 5e-4 --n-bases 40 --testing --gpu 0 --use_cugraph_storage
```
