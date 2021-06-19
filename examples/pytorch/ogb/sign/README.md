SIGN: Scalable Inception Graph Neural Network
==========================
Paper: [https://arxiv.org/abs/2004.11198](https://arxiv.org/abs/2004.11198)


Dependencies
------------
- pytorch 1.5
- dgl 0.5 nightly build
    - `pip install --pre dgl`
- ogb 1.2.3


How to run
-------------
### ogbn-products
```python
python3 sign.py --dataset ogbn-products --eval-ev 10 --R 5 --input-d 0.3 --num-h 512 \
    --dr 0.4 --lr 0.001 --batch-size 50000 --num-runs 10
```

### ogbn-arxiv
```python
python3 sign.py --dataset ogbn-arxiv --eval-ev 10 --R 5 --input-d 0.1 --num-h 512 \
    --dr 0.5 --lr 0.001 --eval-b 100000 --num-runs 10
```

### ogbn-mag
ogbn-mag is a heterogeneous graph and the task is to predict publishing venue
of papers. Since SIGN model is designed for homogeneous graph, we simply ignore
heterogeneous information (i.e. node and edge types) and treat the graph as a
homogeneous one. For node types that don't have input feature, we featurize them
with the average of their neighbors' features.

```python
python3 sign.py --dataset ogbn-mag --eval-ev 10 --R 5 --input-d 0 --num-h 512 \
    --dr 0.5 --lr 0.001 --batch-size 50000 --num-runs 10
```


Results
----------
Table below shows the average and standard deviation (over 10 times) of
accuracy. Experiments were performed on Tesla T4 (15GB) GPU on Oct 29.

| Dataset         | Test Accuracy   | Validation Accuracy   | # Params    |
| :-------------: | :-------------: | :-------------------: | :---------: |
| ogbn-products   | 0.8052±0.0016   | 0.9299±0.0004         | 3,483,703   |
| ogbn-arxiv      | 0.7195±0.0011   | 0.7323±0.0006         | 3,566,128   |
| ogbn-mag        | 0.4046±0.0012   | 0.4068±0.0010         | 3,724,645   |
