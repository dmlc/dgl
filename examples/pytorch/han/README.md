# Heterogeneous Graph Attention Network (HAN) with DGL

This is an attempt to implement HAN with DGL's latest APIs for heterogeneous graphs.
The authors' implementation can be found [here](https://github.com/Jhy1993/HAN).

## Usage

`python main.py` for reproducing HAN's work on their dataset.

`python main.py --hetero` for reproducing HAN's work on DGL's own dataset.

## Performance

Reference performance numbers for the ACM dataset:

|                     | micro f1 score | macro f1 score |
| ------------------- | -------------- | -------------- |
| Paper               | 89.22          | 89.40          |
| DGL                 | 88.99          | 89.02          |
| DGL (own dataset)   | 91.51          | 91.66          |
