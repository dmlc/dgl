Junction Tree VAE - example for training
===

This is a direct modification from https://github.com/wengong-jin/icml18-jtnn

You need to have RDKit installed.

First, download the data directory [here](https://www.dropbox.com/sh/ypxnoqd38kf6ix4/AABKHIZ93DKp1vAoGtJRSj1va?dl=0).  This is the same data folder as the one in original repository.

After that, run
```
python3 molvae/vaetrain_dgl.py -t data/train.txt -v data/vocab.txt
```
