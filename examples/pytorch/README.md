# Model Examples using DGL (w/ Pytorch backend)

Each model is hosted in their own folders. Please read their README.md to see how to
run them.

To understand step-by-step how these models are implemented in DGL. Check out our
[tutorials](https://docs.dgl.ai/tutorials/models/index.html)

## Model summary

Here is a summary of the model accuracy and training speed. Our testbed is Amazon EC2 p3.2x instance (w/ V100 GPU).

| Model | Reported <br> Accuracy | DGL <br> Accuracy | Author's training speed (epoch time) | DGL speed (epoch time) | Improvement |
| ----- | ----------------- | ------------ | ------------------------------------ | ---------------------- | ----------- |
| GCN   | 81.5% | 81.0% | 0.0051s (TF) | 0.0042s | 1.17x |
| TreeLSTM | 51.0% | 51.72% | 14.02s (DyNet) | 3.18s | 4.3x |
| R-GCN <br> (classification) | 73.23% | 73.53% | 0.2853s (Theano) | 0.0273s | 10.4x |
| R-GCN <br> (link prediction) | 0.158 | 0.151 | 2.204s (TF) | 0.633s | 3.5x |
| JTNN | 96.44% | 96.44% | 1826s (Pytorch) | 743s | 2.5x |
| LGNN | 94% | 94% | n/a | 1.45s | n/a |
| DGMG | 84% | 90% | n/a | 1 hr | n/a |
