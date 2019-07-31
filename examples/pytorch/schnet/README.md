SchNet & MGCN
============
- K.T. Schütt. P.-J. Kindermans, H. E. Sauceda, S. Chmiela, A. Tkatchenko, K.-R. Müller.
SchNet: A continuous-filter convolutional neural network for modeling quantum interactions. Advances in Neural Information Processing Systems 30, pp. 992-1002 (2017) [link](http://papers.nips.cc/paper/6700-schnet-a-continuous-filter-convolutional-neural-network-for-modeling-quantum-interactions)
- C. Lu, Q. Liu, C. Wang, Z. Huang, P. Lin, L. He, Molecular Property Prediction: A Multilevel Quantum Interactions Modeling Perspective. The 33rd AAAI Conference on Artificial Intelligence (2019) [link](https://arxiv.org/abs/1906.11081)

Dependencies
------------
- PyTorch 1.0+
- dgl 0.3+
- RDKit (If use Alchemy dataset.)

Usage  
-----

Example usage on Alchemy dataset:

+ SchNet: excepted MAE 0.065
```py
python train.py --model sch --epochs 250
```

+ MGCN: excepted MAE 0.050
```py
python train.py --model mgcn --epochs 250
```

Codes
-----
The folder contains five python files:
- `sch.py` the implementation of SchNet model.
- `mgcn.py` the implementation of Multilevel Graph Convolutional Network(MGCN).
- `layers.py` layers contained by two models above.
- `Alchemy_dataset.py` example dataloader of [Tencent Alchemy](https://alchemy.tencent.com) dataset.
- `train.py` example training code.
Modify `train.py` to switch between different implementations.
