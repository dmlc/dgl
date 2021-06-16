# DGL examples for ogbn-products

Requires DGL 0.5 or later versions.

For the score of `MLP`, run the following command and you should directly see the result.

```bash
python3 mlp.py --eval-last
```

## Results

Here are the results over 10 runs.

| Method | Validation Accuracy |  Test Accuracy  | #Parameters |
|:------:|:-------------------:|:---------------:|:-----------:|
|  MLP   |   0.7841 ± 0.0014   | 0.6320 ± 0.0013 |   535,727   |
