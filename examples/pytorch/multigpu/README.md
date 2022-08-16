Multiple GPU Training
============

Requirements
------------

```bash
pip install torchmetrics
```

How to run
-------

### Graph property predication


Run with following (available dataset: "ogbg-molhiv", "ogbg-molpcba")
```bash
python3 multi_gpu_graph_prediction.py --dataset ogbg-molhiv
```

Results:
```
* ogbg-molhiv: ~0.7965
* ogbg-molpcba: ~0.2239
```

### Node classification
Run with following on dataset "ogbn-products"

```bash
python3 multi_gpu_node_classification.py
```

Results:
```
Test Accuracy: ~0.7632
```