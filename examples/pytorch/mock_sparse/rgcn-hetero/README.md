# Relation-GCN
============

- Paper: [https://arxiv.org/abs/1703.06103](https://arxiv.org/abs/1703.06103)
- Author's code for entity classification: [https://github.com/tkipf/relational-gcn](https://github.com/tkipf/relational-gcn)
- Author's code for link prediction: [https://github.com/MichSchli/RelationPrediction](https://github.com/MichSchli/RelationPrediction)

How to run
-------

Run with the following (available dataset: "aifb", "mutag", "bgs", "am")
```bash
python3 entity_classify.py --dataset aifb
```

Output
-------

Reports training accuracy and loss after each epoch and test accuracy in the end.
```bash
Epoch 00000 | Train Acc: 0.3000 | Train Loss: 7.2665 | Valid Acc: 0.3000 | Valid loss: 7.2665
Epoch 00001 | Train Acc: 0.4571 | Train Loss: 2.1582 | Valid Acc: 0.4571 | Valid loss: 2.1582
Epoch 00002 | Train Acc: 0.7071 | Train Loss: 1.0288 | Valid Acc: 0.7071 | Valid loss: 1.0288
Epoch 00003 | Train Acc: 0.7143 | Train Loss: 0.9438 | Valid Acc: 0.7143 | Valid loss: 0.9438
...
Epoch 00048 | Train Acc: 0.7643 | Train Loss: 0.7254 | Valid Acc: 0.7643 | Valid loss: 0.7254
Epoch 00049 | Train Acc: 0.7643 | Train Loss: 0.7252 | Valid Acc: 0.7643 | Valid loss: 0.7252

Test Acc: 0.7500 | Test loss: 0.7506
```
