Community Detection with Graph Neural Networks (CDGNN)
============

Paper link: [https://openreview.net/pdf?id=H1g0Z3A9Fm](https://openreview.net/pdf?id=H1g0Z3A9Fm)

Author's code repo: [https://github.com/zhengdao-chen/GNN4CD](https://github.com/zhengdao-chen/GNN4CD)

This folder contains a DGL implementation of the CDGNN model.

Dependencies
--------------
* PyTorch 0.4.1+
* requests

```bash
pip install torch requests
```

How to run
----------

An experiment on the Stochastic Block Model in default settings can be run with

```bash
python3 train.py
```

An experiment on the Stochastic Block Model in customized settings can be run with
```bash
python3 train.py --batch-size BATCH_SIZE --gpu GPU --n-communities N_COMMUNITIES \
                --n-features N_FEATURES --n-graphs N_GRAPH --n-iterations N_ITERATIONS \
                --n-layers N_LAYER --n-nodes N_NODE --model-path MODEL_PATH --radius RADIUS
```
