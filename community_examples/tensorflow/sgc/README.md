# Simple Graph Convolution (SGC)

> Graph Convolutional Networks derive inspiration primarily from recent deep learning approaches, and as a result, may inherit unnecessary complexity and redundant computation. In this paper, we reduce this excess complexity through successively removing nonlinearities and collapsing weight matrices between consecutive layers. We theoretically analyze the resulting linear model and show that it corresponds to a fixed low-pass filter followed by a linear classifier.

* [Paper](https://arxiv.org/abs/1902.07153)
* [Author Implementation](https://github.com/Tiiiger/SGC)

Note: TensorFlow uses a different implementation of weight decay in AdamW to PyTorch. This results in differences in performance. You can see this by manually adding the L2 of the weights to the loss like [this](https://github.com/dmlc/dgl/blob/d696558b0bbcb60f1c4cf68dc93cd22c1077ce06/examples/tensorflow/gcn/train.py#L99) for comparison.

## Requirements

This example is tested with TensorFlow 2.3.0.

```bash
$ pip install dgl tensorflow tensorflow_addons
```

## Usage
```bash
$ python sgc.py --help
usage: sgc.py [-h] [--dataset DATASET] [--lr LR] [--bias]
              [--n-epochs N_EPOCHS] [--weight-decay WEIGHT_DECAY]

Run experiment for Simple Graph Convolution (SGC)

optional arguments:
  -h, --help                    show this help message and exit
  --dataset DATASET             dataset to run
  --lr LR                       learning rate
  --bias                        flag to use bias
  --n-epochs N_EPOCHS           number of training epochs
  --weight-decay WEIGHT_DECAY   weight for L2 loss
```

## Results
```bash
# Cora citation network dataset
$ python sgc.py --dataset cora --lr 0.2 --n-epochs 100 --weight-decay 5e-6
...
Epoch 100/100
1/1 [==============================] - 0s 40ms/step - loss: 0.0313 - accuracy: 1.0000 - val_loss: 0.7870 - val_accuracy: 0.7620
Test Accuracy: 77.2%

# Citeseer citation network dataset
$ python sgc.py --dataset citeseer --lr 0.2 --n-epochs 150 --bias --weight-decay 5e-5
...
Epoch 150/150
1/1 [==============================] - 0s 65ms/step - loss: 0.0160 - accuracy: 1.0000 - val_loss: 1.1021 - val_accuracy: 0.6420
Test Accuracy: 63.9%

# Pubmed citation network dataset
$ python sgc.py --dataset pubmed --lr 0.2 --n-epochs 100 --bias --weight-decay 5e-5
...
Epoch 100/100
1/1 [==============================] - 0s 52ms/step - loss: 0.0421 - accuracy: 1.0000 - val_loss: 0.5862 - val_accuracy: 0.7680
Test Accuracy: 76.3%
```

| Dataset  | Accuracy | Paper |
|----------|----------|-------|
| Cora     | 77.3%    | 81.0% |
| Citeseer | 63.9%    | 71.9% |
| Pubmed   | 76.4%    | 78.9% |
