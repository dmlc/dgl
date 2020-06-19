# Inductive Graph-based Matrix Completion

Paper link: [https://arxiv.org/abs/1904.12058](https://arxiv.org/abs/1904.12058)
Author's code: [https://github.com/muhanzhang/IGMC](https://github.com/muhanzhang/IGMC)

Credit: Tianjun Xiao ([@sneakerkg](https://github.com/sneakerkg))

## Dependencies

* PyTorch 1.2+
* pandas
* torchtext 0.4+ (if using user and item contents as node features)

## Data

Supported datasets: ml-100k, ml-1m, ml-10m

## How to run

### Train with full-graph

ml-100k, no feature

```shell
python3 train.py --data_name=ml-100k --use_one_hot_fea --train_max_epoch 80 --max_nodes_per_hop 200 --batch_size 50 --train_decay_epoch 50 --num_igmc_bases 4  --train_min_lr 1e-6 --data_valid_ratio 0.2
```

### How to run ensemble evaluation

```shell
python3 train.py --data_name=ml-100k --use_one_hot_fea --train_max_epoch 80 --batch_size 50 --train_decay_epoch 50 --num_igmc_bases 4  --ckpt_idxs 39,49,59,69,79
```
