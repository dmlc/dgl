# directional_GSN

## Introduction

This is an example of implementing [directional_GSN](https://arxiv.org/abs/2006.09252) for graph classification in DGL.

directional_GSN is a combination of GSN with DGN, where we defined a vector field based on structural features instead o f eigenvectors.

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

## Usage


run the scripts in `./scripts` folder
```{.bash}
scripts/run_molpcba.sh
```
and you can modify the arguments of the scripts.

## Experiments

We fix the random seed to 41, and take over 10 runs for the model. The model is trained on a single Tesla T4 GPU with 16GB memory.

### ogbg-molpcba

#### performance

|                 | train_AP | valid_AP | test_AP | #parameters |
| --------------- | ---------| -------- | ------- | ----------- |
| directional_GSN |                               | 5142713     |


#### Reproduction of performance

```{.bash}
python main.py --device 0 --epochs 450 --dropout 0.2
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