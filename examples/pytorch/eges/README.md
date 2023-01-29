# DGL & Pytorch implementation of Enhanced Graph Embedding with Side information (EGES)
Paper link: https://arxiv.org/pdf/1803.02349.pdf
Reference code repo: (https://github.com/wangzhegeek/EGES.git)

## How to run

- Create a folder named `data`.
`mkdir data`
- Download csv data
`wget https://raw.githubusercontent.com/Wang-Yu-Qing/dgl_data/master/eges_data/action_head.csv -P data/`
`wget https://raw.githubusercontent.com/Wang-Yu-Qing/dgl_data/master/eges_data/jdata_product.csv -P data/`
- Run with the following command (with default configuration)
`python main.py`

## Result
```
Evaluate link prediction AUC: 0.7084
```
