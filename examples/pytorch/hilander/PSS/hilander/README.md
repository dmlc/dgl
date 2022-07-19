# hilander

Referenced from the ICCV '22 paper ["Learning Hierarchical Graph Neural Networks for Image Clustering"](https://arxiv.org/abs/2107.01319), code is from [hilander code](https://github.com/dmlc/dgl/tree/65ecbb94921aa961a9be643100262c9abafc1830/examples/pytorch/hilander).

## Data
Training data are generated from feature extractor Smooth_AP.

## Training

For training, you can run `bash scripts/train_inat.sh`:

```commandline
CUDA_VISIBLE_DEVICES=0 python train_subg.py \
--data_path '/home/ubuntu/code/Smooth_AP/data/T_train_iter0_smoothap_inat_features.pkl' \
--model_filename 'checkpoint/inat_l_smoothap_iter1.pth' \
--knn_k 10,5,3 --levels 2,3,4 \
--hidden 512 --epochs 1000 --lr 0.01 \
--batch_size 4096 --num_conv 1 --gat --balance
```

* `--data_path`: data path of the feature pickle file;
* `--model_filename`: model save path;

## Inference

For inference, you can run `bash scripts/test_inat.sh`:

```commandline
CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 python test_subg_inat.py \
--data_path '/home/ubuntu/code/Smooth_AP/data/all_train_iter1_smoothap_inat_features.pkl' \
--model_filename 'checkpoint/inat_l_smoothap_iter1.pth'  --knn_k 10 \
--tau 0.9 --level 10 --threshold prob \
--hidden 512 --num_conv 1 --gat --batch_size 4096 --early_stop \
--mode selectbydensity --thresh 0.8 \
--linsize 29011 --uinsize 18403 --inclasses 948 \
--output_filename 'data/inat_hilander_l_smoothap_train_selectbydensity_0.8_0.9_iter1.pkl'
```

* `--data_path`: all training data pickle file path;
* `--model_filename`: hilander model file path;
* `--mode`: test mode, select from ['selectbydensity', 'recluster', 'donothing'];
* `--thresh`: density selection threshold;
* `--output_filename`: features with masks and pseudo labels pickle file path.

To draw the density-Lin-distance and save the distance pickle file, add `--draw true`.