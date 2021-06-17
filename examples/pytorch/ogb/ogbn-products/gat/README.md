# DGL examples for ogbn-products

## Sample-based GAT

Requires DGL 0.4.3post2 or later versions.

Run `main.py` and you should directly see the result.

Accuracy over 5 runs: 0.7863197 ± 0.00072568655

## GAT (another implementation)

Requires DGL 0.5 or later versions.

For the score of `GAT`, run the following command and you should directly see the result.

```bash
python3 gat.py
```

Or, if you want to speed up during training time, run with `--estimation-mode` enabled.
This option will do a complete evaluation when the training is over.

```bash
python3 gat.py --estimation-mode
```

## Results

Here are the results over 10 runs.

|    Method     | Validation Accuracy |  Test Accuracy  | #Parameters |
|:-------------:|:-------------------:|:---------------:|:-----------:|
| GAT (main.py) |         N/A         | 0.7863 ± 0.0007 |     N/A     |
| GAT (gat.py)  |   0.9327 ± 0.0003   | 0.8126 ± 0.0018 |  1,065,127  |
