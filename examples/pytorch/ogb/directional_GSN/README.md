# directional_GSN

## Introduction

This is an example of implementing [directional_GSN](https://arxiv.org/abs/2006.09252) for graph classification in DGL.

directional_GSN is a combination of Graph Substructure Networks ([GSN](https://arxiv.org/abs/2006.09252)) with Directional Graph Networks ([DGN](https://arxiv.org/pdf/2010.02863.pdf)), where we defined a vector field based on substructure encoding instead of Laplacian eigenvectors.

The script in this folder experiments directional_GSN on ogbg-molpcba dataset.

## Installation requirements
```
conda create --name gsn python=3.7
conda activate gsn
conda install pytorch==1.11.0 cudatoolkit=10.2 -c pytorch
pip install tqdm
pip install networkx
conda install -c conda-forge graph-tool
pip install ogb
pip install dgl-cu102 -f https://data.dgl.ai/wheels/repo.html
```

## Experiments

We fix the random seed to 41, and train the model on a single Tesla T4 GPU with 16GB memory.

### ogbg-molpcba

#### performance

|                  | train_AP | valid_AP | test_AP | #parameters |
| ---------------- | ---------| -------- | ------- | ----------- |
| directional_GSN  | 0.4301   | 0.2598   | 0.2438  | 5142713     |


#### Reproduction of performance

```{.bash}
python preprocessing.py
python main.py --seed 41 --epochs 450 --hidden_dim 420 --out_dim 420 --dropout 0.2
```

## References

```{.tex}
@article{bouritsas2020improving,
  title={Improving graph neural network expressivity via subgraph isomorphism counting},
  author={Bouritsas, Giorgos and 
          Frasca, Fabrizio and 
          Zafeiriou, Stefanos and 
          Bronstein, Michael M},
  journal={arXiv preprint arXiv:2006.09252},
  year={2020}
}
```