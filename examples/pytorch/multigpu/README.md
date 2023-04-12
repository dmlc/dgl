Multiple GPU Training
============

Requirements
------------

```bash
pip install torchmetrics==0.11.4
```

How to run
-------

### Graph property prediction


Run with following (available dataset: "ogbg-molhiv", "ogbg-molpcba")
```bash
python3 multi_gpu_graph_prediction.py --dataset ogbg-molhiv
```

#### __Results__
```
* ogbg-molhiv: ~0.7965
* ogbg-molpcba: ~0.2239
```

#### __Scalability__
We test scalability of the code with dataset "ogbg-molhiv" in a machine of type <a href="https://aws.amazon.com/blogs/aws/now-available-ec2-instances-g4-with-nvidia-t4-tensor-core-gpus/">Amazon EC2 g4dn.metal</a>
, which has **8 Nvidia T4 Tensor Core GPUs**.


|GPU number |Speed Up |Batch size |Test accuracy |Average epoch Time|
| --- | ----------- | ----------- | -----------|-----------|
| 1 | x | 32 | 0.7765| 45.0s|
| 2 | 3.7x |64 | 0.7761|12.1s|
| 4 | 5.9x| 128 |  0.7854|7.6s|
| 8 | 9.5x| 256 |  0.7751|4.7s|


### Node classification


Run with following on dataset "ogbn-products"

```bash
python3 multi_gpu_node_classification.py
```

#### __Results__
```
Test Accuracy: ~0.7632
```

### Link prediction


Run with following (available dataset: "ogbn-products", "reddit")

```bash
python3 multi_gpu_link_prediction.py --dataset ogbn-products
```

#### __Results__
```
Eval F1-score: ~0.7999  Test F1-score: ~0.6383
```

Notably,

* The loss function is defined by predicting whether an edge exists between two nodes or not.
* When computing the score of `(u, v)`, the connections between node `u` and `v` are removed from neighbor sampling.
* The performance of the learned embeddings are measured by training a softmax regression with scikit-learn.
