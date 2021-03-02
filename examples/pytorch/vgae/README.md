# Variational Graph Auto-Encoders

- Paper link：https://arxiv.org/abs/1611.07308
- Author's code repo：https://github.com/tkipf/gae

## Requirements

- Pytorch
- Python 3.x 
- DGL 0.6
- scikit-learn

## Run the demo

Run with following (available dataset: "cora", "citeseer", "pubmed")

```
python train.py
```

## Dataset

In this example, I use two kinds of data source. One from DGL's bulit-in dataset (CoraGraphDataset, CiteseerGraphDataset and PubmedGraphDataset), another from website https://github.com/kimiyoung/planetoid.

You can specify a dataset as follows:

```
python train.py --datasrc dgl --dataset cora  // from DGL
python train.py --datasrc website --dataset cora  // from website
```

**Note**: If you want to train by dataset from website, you should download folder https://github.com/kimiyoung/planetoid/tree/master/data. Then put it under project folder.

## Results

Use *area under the ROC curve* (AUC) and *average precision* (AP) scores for each model on the test set. Numbers show mean results and standard error for 10 runs with random initializations on fixed dataset splits.

### Dataset from DGL

| Dataset  | AUC            | AP            |
| -------- | -------------- | ------------- |
| Cora     | 91.8$\pm$ 0.01 | 92.5$\pm$0.01 |
| Citeseer | 89.2$\pm$0.02  | 90.8$\pm$0.01 |
| Pubmed   | 94.5$\pm$0.01  | 94.6$\pm$0.01 |

### Dataset from website

| Dataset  | AUC            | AP             |
| -------- | -------------- | -------------- |
| Cora     | 90.9$\pm$ 0.01 | 92.1$\pm$0.01  |
| Citeseer | 90.3$\pm$0.01  | 91.8$\pm$0.01  |
| Pubmed   | 94.4$\pm$ 0.01 | 94.6$\pm$ 0.01 |

### Reported results in paper

| Dataset  | AUC            | AP            |
| -------- | -------------- | ------------- |
| Cora     | 91.4$\pm$ 0.01 | 92.6$\pm$0.01 |
| Citeseer | 90.8$\pm$0.02  | 92.0$\pm$0.02 |
| Pubmed   | 94.4$\pm$0.02  | 94.7$\pm$0.02 |

