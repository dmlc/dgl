# Graph Convolutional Matrix Completion

Paper link: [https://arxiv.org/abs/1706.02263](https://arxiv.org/abs/1706.02263)
Author's code: [https://github.com/riannevdberg/gc-mc](https://github.com/riannevdberg/gc-mc)

The implementation does not handle side-channel features and mini-epoching and thus achieves
slightly worse performance when using node features.

Credit: Jiani Zhang ([@jennyzhang0215](https://github.com/jennyzhang0215))

## Dependencies
* PyTorch 1.2+
* pandas
* torchtext 0.4+ (if using user and item contents as node features)

## Data

Supported datasets: ml-100k, ml-1m, ml-10m

## How to run
### Train with full-graph
ml-100k, no feature
```bash
python3 train.py --data_name=ml-100k --use_one_hot_fea --train_max_epoch 80 --max_nodes_per_hop 200 --save_dir /log/dir/ --batch_size 50 --train_decay_epoch 50 --num_igmc_bases 4  --train_min_lr 1e-6 --data_valid_ratio 0.2

```
Results: RMSE=0.9088 (0.905 reported)
Speed: 0.0410s/epoch (vanilla implementation: 0.1008s/epoch)

### How to run ensemble evaluation
```bash
python3 train.py --data_name=ml-100k --use_one_hot_fea --train_max_epoch 80 --save_dir /log/dir/ --batch_size 50 --train_decay_epoch 50 --num_igmc_bases 4  --ckpt_idxs 39,49,59,69,79

```