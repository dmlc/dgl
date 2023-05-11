# Multiple GPU Training

## Requirements

```bash
pip install torchmetrics==0.11.4
```

## How to run

### Node classification

Run with following (available dataset: "ogbn-products", "ogbn-arxiv")

```bash
python3 node_classification_sage.py --dataset_name ogbn-products
```

#### __Results__ with default arguments
```
* Test Accuracy of "ogbn-products": ~0.7716
* Test Accuracy of "ogbn-arxiv": ~0.6994
```
