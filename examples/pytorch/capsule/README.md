DGL implementation of Capsule Network
=====================================

This repo implements Hinton and his team's [Capsule Network](https://arxiv.org/abs/1710.09829).
Only margin loss is implemented, for simplicity to understand the DGL.

Dependencies
--------------
* PyTorch 0.4.1+
* torchvision

```bash
pip install torch torchvision
```

Training & Evaluation
----------------------
```bash
# Run with default config
python3 main.py
# Run with train and test batch size 128, and for 50 epochs
python3 main.py --batch-size 128 --test-batch-size 128 --epochs 50
```
