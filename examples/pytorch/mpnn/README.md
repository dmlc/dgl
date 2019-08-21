MPNN
============

Message Passing Neural Network

Gilmer, Justin, et al. "Neural message passing for quantum chemistry." Proceedings of the 34th International Conference on Machine Learning-Volume 70. JMLR. org, 2017. [link](https://arxiv.org/pdf/1704.01212.pdf)

Thanks [@MilkshakeForReal](https://discuss.dgl.ai/t/mpnn-implementation/356) for providing the draft-version implementation.

Dependencies
------------
- PyTorch 1.0+
- dgl 0.3+
- RDKit (If use [Alchemy dataset](https://arxiv.org/abs/1906.09427).)

Usage  
------------

Example usage on Alchemy dataset:

Expected MAE 0.092
```py
python train.py --epochs 250
```

*With Tesla V100, MPNN takes 240s/epoch.*

Codes
-----
The folder contains three python files:
- `mpnn.py` the implementation of MPNN.
- `Alchemy_dataset.py` example dataloader of [Tencent Alchemy](https://alchemy.tencent.com) dataset.
- `train.py` example training code.
Modify `train.py` to switch between different configurations.
