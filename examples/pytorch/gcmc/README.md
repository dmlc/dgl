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
### Train with full-graph

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

### Train with minibatch
ml-100k, no feature
```bash
python gcmc_sampling.py --data_name=ml-100k --use_one_hot_fea --gcn_agg_accum=stack
```
ml-100k, no feature with mix_cpu_gpu run
```bash
python gcmc_sampling.py --data_name=ml-100k --use_one_hot_fea --gcn_agg_accum=stack  --mix_cpu_gpu
```
Results: RMSE=0.9088 (0.910 reported)
Speed: 0.385s/epoch (Run with only 200 epoches instead of 2000)
Speed: 0.500s/epoch (mix_cpu_gpu)

ml-100k, with feature
```bash
python gcmc_sampling.py --data_name=ml-100k --gcn_agg_accum=stack
```
Results: RMSE=0.9568 (0.905 reported)

ml-1m, no feature
```bash
python3 gcmc_sampling.py --data_name=ml-1m --gcn_agg_accum=sum  --use_one_hot_fea
```
ml-1m, no feature with mix_cpu_gpu run
```bash
python3 gcmc_sampling.py --data_name=ml-1m --gcn_agg_accum=sum --use_one_hot_fea --mix_cpu_gpu
```
Results: RMSE=0.8387 (0.832 reported)
Speed: 18.279s/epoch (Run with only 200 epoches instead of 2000)
Speed: 32.567s/epoch (mix_cpu_gpu)

ml-10m, no feature
```bash
python gcmc_sampling.py --data_name=ml-10m --gcn_agg_accum=stack --gcn_dropout=0.3 \
                                 --train_lr=0.001 --train_min_lr=0.0001 --train_max_epoch=60 \
                                 --use_one_hot_fea --gen_r_num_basis_func=4
```
ml-10m, no feature with mix_cpu_gpu run
```bash
python gcmc_sampling.py --data_name=ml-10m --gcn_agg_accum=stack --gcn_dropout=0.3 \
                                 --train_lr=0.001 --train_min_lr=0.0001 --train_max_epoch=60 \
                                 --use_one_hot_fea --gen_r_num_basis_func=4 --mix_cpu_gpu
```
Results: RMSE=0.7809 (0.777 reported)
Speed: 1414.38s/epoch (Run with only 60 epoches instead of 15000)
Speed: 1746.97s/epoch (mix_cpu_gpu)
Testbed: EC2 p3.2xlarge instance(Amazon Linux 2)


### Train with minibatch and multigpu
ml-100k, with feature
```bash
python gcmc_mp_sampling.py --data_name=ml-100k --gcn_agg_accum=stack --gpu 0,1,2,3,4,5,6,7
```