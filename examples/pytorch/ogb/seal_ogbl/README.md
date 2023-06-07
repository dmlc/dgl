# SEAL Implementation for OGBL in DGL

Introduction
------------
This is an example of implementing [SEAL](https://arxiv.org/pdf/2010.16103.pdf) for link prediction in DGL. Some parts are migrated from [https://github.com/facebookresearch/SEAL_OGB](https://github.com/facebookresearch/SEAL_OGB).

Requirements
------------
[PyTorch](https://pytorch.org/), [DGL](https://www.dgl.ai/), [OGB](https://ogb.stanford.edu/docs/home/), and other python libraries: numpy, scipy, tqdm, scikit-learn, etc.

Usages
------
Run the following command for results on each benchmark
```bash
# ogbl-ppa
python main.py \
    --dataset ogbl-ppa \
    --use_feature \
    --use_edge_weight \
    --eval_steps 5 \
    --epochs 20 \
    --train_percent 5 

# ogbl-collab
python main.py \
    --dataset ogbl-collab \
    --train_percent 15 \
    --hidden_channels 256 \
    --use_valedges_as_input

# ogbl-ddi
python main.py \
    --dataset ogbl-ddi \
    --ratio_per_hop 0.2 \
    --use_edge_weight \
    --eval_steps 1 \
    --epochs 10 \
    --train_percent 5

# ogbl-citation2
python main.py \
    --dataset ogbl-citation2 \
    --use_feature \
    --use_edge_weight \
    --eval_steps 1 \
    --epochs 10 \
    --train_percent 2 \
    --val_percent 1 \
    --test_percent 1
```

Results
-------

|              | ogbl-ppa (Hits@100) | ogbl-collab (Hits@50) | ogbl-ddi (Hits@20) | ogbl-citation2 (MRRd) |
|--------------|---------------------|-----------------------|--------------------|---------------------|
| Paper Test Results |  48.80%&plusmn;3.16% |    64.74%&plusmn;0.43% | 30.56%&plusmn;3.86%* |   87.67%&plusmn;0.32r% |
| Our Test Results |  49.48%&plusmn;2.52% |    64.23%&plusmn;0.57% | 27.93%&plusmn;4.19% |   86.29%&plusmn;0.47% |

\* Note that the relatively large gap on ogbl-ddi may come from the high variance of results on this dataset. We get 28.77%&plusmn;3.43% by only changing the sampling seed.

Reference
---------

    @article{zhang2021labeling,
        title={Labeling Trick: A Theory of Using Graph Neural Networks for Multi-Node Representation Learning},
        author={Zhang, Muhan and Li, Pan and Xia, Yinglong and Wang, Kai and Jin, Long},
        journal={Advances in Neural Information Processing Systems},
        volume={34},
        year={2021}
        }

    @inproceedings{zhang2018link,
      title={Link prediction based on graph neural networks},
      author={Zhang, Muhan and Chen, Yixin},
      booktitle={Advances in Neural Information Processing Systems},
      pages={5165--5175},
      year={2018}
    }
