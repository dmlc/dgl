# Graph Convolutional Matrix Completion

Paper link: [https://arxiv.org/abs/1706.02263](https://arxiv.org/abs/1706.02263)
Author's code: [https://github.com/riannevdberg/gc-mc](https://github.com/riannevdberg/gc-mc)

The implementation does not handle side-channel features and mini-epoching and thus achieves
slightly worse performance when using node features.

Credit: Jiani Zhang ([@jennyzhang0215](https://github.com/jennyzhang0215))

## Dependencies
* PyTorch 1.2+
* pandas
* torchtext 0.4+

## Data

Supported datasets: ml-100k, ml-1m, ml-10m

## How to run

ml-100k, no feature
```bash
python train.py --data_name=ml-100k --use_one_hot_fea --gcn_agg_accum=stack
```
Results: RMSE=0.9088 (0.910 reported)
Speed: 0.0195s/epoch (vanilla implementation: 0.1008s/epoch)

ml-100k, with feature
```bash
python train.py --data_name=ml-100k --gcn_agg_accum=stack
```
Results: RMSE=0.9448 (0.905 reported)

ml-1m, no feature
```bash
python train.py --data_name=ml-1m --gcn_agg_accum=sum --use_one_hot_fea
```
Results: RMSE=0.8377 (0.832 reported)
Speed: 0.0557s/epoch (vanilla implementation: 1.538s/epoch)

ml-10m, no feature
```bash
python train.py --data_name=ml-10m --gcn_agg_accum=stack --gcn_dropout=0.3 \
                                 --train_lr=0.001 --train_min_lr=0.0001 --train_max_iter=15000 \
                                 --use_one_hot_fea --gen_r_num_basis_func=4
```
Results: RMSE=0.7800 (0.777 reported)
Speed: 0.9207/epoch (vanilla implementation: OOM)

Testbed: EC2 p3.2xlarge instance(Amazon Linux 2)