Spatio-Temporal Graph Convolutional Networks
============

- Paper link: [arXiv](https://arxiv.org/pdf/1709.04875v4.pdf)
- Author's code repo: [https://github.com/VeritasYin/STGCN_IJCAI-18](https://github.com/VeritasYin/STGCN_IJCAI-18).
Dependencies
------------
- PyTorch 1.1.0+
- sklearn
- dgl



How to run
----------

An experiment in default settings can be run with

```bash
python main.py
```

An experiment on the GIN in customized settings can be run with
```bash
python main.py --lr --seed --disable-cuda --batch_size  --epochs
```

Results
-------

```bash
python main.py
```
METR_LA MAE: ~5.76