# GraphParticleSim
## DGL Implementation of Learning-to-simulate paper.

This DGL example implements the GNN model proposed in the paper [Learning-to-simulate](https://arxiv.org/abs/2002.09405.pdf). 

GraphParticleSim implementor
----------------------
This example was implemented by [Ericcsr](https://github.com/Ericcsr) during his Internship work at the AWS Shanghai AI Lab.

The graph dataset used in this example 
---------------------------------------
This Example uses Datasets Generate By Taichi-MPM Simulator

MPM2d:
    - 2048 Particles/Nodes
    - Edges are determined Dynamically by radius r nearest neighbor in 2d
    - 100 trajectories should be generated
    - 400 steps of simulation per trajectory

MPM3d:
    - 8192 Particles/Nodes
    - Edges are determined Dynamically by radius r nearest neighbor in 3d
    - 100 trajectories should be generated
    - 400 steps of simulation per trajectory

Dependency
--------------------------------
```
pip install taichi
```


How to run example files
--------------------------------
In the graphsim folder, run
**Please first run `data_gen.py`**

Using Ground Truth Velocity From Simulator Directly.

```python
python data_gen.py --num_traj 100 --steps 400 --order first
```

Using Finite Difference Velocity from position.

```python
python data_gen.py --num_traj 100 --steps 400 --order second
```

**Please use `train_datasets.py`**


```python
python train_datasets.py --gpu 0 --number_workers 15 --radius 0.015
```

It is highly suggested to run on GPU due to the gigantic simulation graph,if you want to run on CPU

```python
python train_datasets.py --gpu -1 --number_workers 15 --radius 0.015
```

One Step Loss Performance
-------------------------
| Models/Dataset | Water |
| :-------------- | -----: |
| GraphSim in DGL | 81.5% |
| GraphSim | 81.8% |

Speed Performance Time/Steps
-------------------------
| Models/Dataset | Water |
| :-------------- | -----: |
| GraphSim in DGL | |

Notice that The datasets are generate dataset directly from Taichi Simulator to prevent using Tensorflow to handle the original dataset.

