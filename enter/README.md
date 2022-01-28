# DGL-Enter Guide

## Installation guide
`pip install dglenter`


## Usage guide

DGL enter is a new tool for user to bootstrap datasets and common models.

The entry point of enter is `dgl-enter`, and it has three subcommand `config`, `train` and `export`.

How to train a model on a dataset with task:
1. Run `dgl-enter config` to generate a configuration file
2. Run `dgl-enter train` to start a training script base on the configuration file generated. Or you can use `dgl-enter export` to export a self-contained, runnable python script based on your configuration file




`dgl-ente` currently provides 4 pipelines:
- nodepred (Node prediction tasks, suitable for small dataset to prototype)
- nodepred-ns (Node prediction tasks with sampling method, suitable for medium and large dataset)
- edgepred (Link prediction tasks, to predict whether edge exists among node pairs based on node features)
- graphpred (Graph prediction tasks, supervised graph classification tasks)

You can get the full list by `dgl-enter config --help` (TODO: update when graph pred is added)
```
Usage: dgl-enter config [OPTIONS] COMMAND [ARGS]...

  Generate the config files

Options:
  --help  Show this message and exit.

Commands:
  edgepred     Edge classification pipeline
  nodepred     Node classification pipeline
  nodepred-ns  Node classification sampling pipeline
```

For each pipeline it will have diffirent options to specified. For example, for node prediction pipeline, you can do `dgl-enter config nodepred --help`, you'll get:
```
Usage: dgl-enter config nodepred [OPTIONS]

  Node classification pipeline

Options:
  --data [cora|citeseer|ogbl-collab|csv|reddit|co-buy-computer]
                                  input data name  [required]
  --cfg TEXT                      output configuration path  [default:
                                  cfg.yml]
  --model [gcn|gat|sage|sgc|gin]  Model name  [required]
  --device [cpu|cuda]             Device, cpu or cuda  [default: cpu]
  --help                          Show this message and exit.
```

For example, you can do `dgl-enter config nodepred --data cora --model gcn --cfg cfg.yml` to initialize a configuration for node prediction tasks on Cora graph dataset with the graph convolutional network (GCN). The configuration file is like below:`
```yaml
version: 0.0.1
pipeline_name: nodepred
device: cpu
data:
  name: cora
model:
  name: gcn
  hidden_size: 16             # Hidden size.
  num_layers: 1               # Number of layers.
  norm: both                  # GCN normalization type. Can be 'both', 'right', 'left', 'none'.
  activation: relu            # Activation function.
  dropout: 0.5                # Dropout rate.
  use_edge_weight: false      # If true, scale the messages by edge weights.
general_pipeline:
  node_embed_size: -1         # The node learnable embedding size, -1 to disable
  early_stop:
    patience: 20              # Steps before early stop
    checkpoint_path: checkpoint.pth # Early stop checkpoint model file path
  num_epochs: 200             # Number of training epochs
  eval_period: 5              # Interval epochs between evaluations
  optimizer:
    name: Adam
    lr: 0.005
  loss: CrossEntropyLoss
```

You can tweak the numbers and the options here as you wish, to try diffirent settings and get better performance.

Then you can use `dgl-enter train --cfg cfg.yml` to train a model on the dataset specified. Or you can use `dgl-enter export --cfg cfg.yml --output run.py` to generate a self-contained, runnable python script. Then you can do any modification on the script as you like to adapt into your own pipeline