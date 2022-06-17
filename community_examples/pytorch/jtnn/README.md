Junction Tree VAE - example for training
==========================================

This is a direct modification from https://github.com/wengong-jin/icml18-jtnn

Dependencies
--------------
* PyTorch 0.4.1+
* RDKit=2018.09.3.0
* requests

How to run
-----------

To run the model, use
```
python3 vaetrain_dgl.py
```
The script will automatically download the data, which is the same as the one in the
original repository.

To disable CUDA, run with `NOCUDA` variable set:
```
NOCUDA=1 python3 vaetrain_dgl.py
```

To decode for new molecules, run
```
python3 vaetrain_dgl.py -T
```

Currently, decoding involves encoding a training example, sampling from the posterior
distribution, and decoding a molecule from that.
