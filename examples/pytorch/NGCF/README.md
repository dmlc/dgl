# DGL Implementation of the NGCF Model

This DGL example implements the GNN model proposed in the paper [Neural Graph Collaborative Filtering](https://arxiv.org/abs/1905.08108). 
The author's codes of implementation is in [here](https://github.com/xiangwang1223/neural_graph_collaborative_filtering). A pytorch re-implementation can be found [here](https://github.com/huangtinglin/NGCF-PyTorch).

Example implementor
----------------------
This example was implemented by [Kounianhua Du](https://github.com/KounianhuaDu) during her Software Dev Engineer Intern work at the AWS Shanghai AI Lab.


The graph dataset used in this example 
---------------------------------------
Gowalla: This is the check-in dataset obtained from Gowalla, where users share their locations by checking-in. To ensure the quality of the dataset, we use the 10-core setting, i.e., retaining users and items with at least ten interactions. The dataset used can be found [here](https://github.com/xiangwang1223/neural_graph_collaborative_filtering/tree/master/Data).

Statistics:
- Users: 29858
- Items: 40981
- Interactions: 1027370
- Density: 0.00084


How to run example files
--------------------------------
First to get the data, in the Data folder, run

```bash
sh load_gowalla.sh
```

Then, in the NGCF folder, run

```bash
python main.py --dataset gowalla --regs [1e-5] --embed_size 64 --layer_size [64,64,64] --lr 0.0001 --save_flag 1 --batch_size 1024 --epoch 400 --verbose 1 --mess_dropout [0.1,0.1,0.1] --gpu 0 
```

NOTE: Following the paper's setting, the node dropout is disabled.


Performance
-------------------------
The following results are the results in 400 epoches.

**NGCF results**
| Model         | Paper (tensorflow)               | ours (DGL)                  |
| ------------- | -------------------------------- | --------------------------- |
| recall@20     | 0.1569                           | 0.1552                      |
| ndcg@20       | 0.1327                           | 0.2707                      |

