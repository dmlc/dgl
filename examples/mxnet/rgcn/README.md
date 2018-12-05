# Relational-GCN

### Prerequisites
Two extra python packages are needed for this example: 

- rdflib
- pandas

Example code was tested with rdflib 4.2.2 and pandas 0.23.4

### Entity Classification
AIFB:
```
DGLBACKEND=mxnet python mxnet/rgcn/entity_classify.py -d aifb --testing --gpu 0
```

MUTAG:
```
DGLBACKEND=mxnet python mxnet/rgcn/entity_classify.py -d mutag --l2norm 5e-4 --n-bases 40 --testing --gpu 0
```

BGS:
```
DGLBACKEND=mxnet python mxnet/rgcn/entity_classify.py -d bgs --l2norm 5e-4 --n-bases 20 --testing --gpu 0 --relabel
```
