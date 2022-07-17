# Smooth_AP

Referenced from the ECCV '20 paper ["Smooth-AP: Smoothing the Path Towards Large-Scale Image Retrieval"](https://www.robots.ox.ac.uk/~vgg/research/smooth-ap/), code is from https://github.com/Andrew-Brown1/Smooth_AP.

The PyTorch implementation of the Smooth-AP loss function is found in src/Smooth_AP_loss.py

![teaser](https://github.com/Andrew-Brown1/Smooth_AP/blob/master/ims/teaser.png)

## Data

This repository is used for training using Smooth-AP loss on the following datasets:
INaturalist (2018 version - obtained from this website https://www.kaggle.com/c/inaturalist-2018/data)

The annotation txt files for different data splits are in `data/inaturalist`.

## Training the model

### iteration 0 training

Training on the iNaturalist dataset in iteration 0, you can run `scripts/train_inat.sh`:

```commandline
CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 python src/main.py \
--dataset Inaturalist --lr 1e-5 --fc_lr_mul 1 \
--n_epochs 400 --bs 384 \
--source_path ~/code/Smooth_AP/data/ --embed_dim 128 \
--class_num 948 --loss smoothap --infrequent_eval 1
```

* `--loss`: training loss, selected from `['smoothap', 'ams', 'ce']`;
* `--class_num`: class number, which is the fc output layer dimension. only effective with loss ams and ce.

### iteration k training (k > 0)

Training on the iNaturalist dataset in iterations after iteration 0, you can run `scripts/finetune_inat_1head.sh`:

```commandline
CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 python src/finetune_1head.py \
--dataset Inaturalist --lr 1e-5 --fc_lr_mul 1 \
--n_epochs 400 --bs 384 --class_num 1024 \
--source_path ~/code/Smooth_AP/data/ --embed_dim 128 \
--trainset lin_train_set1.txt --testset Inaturalist_test_set1.txt \
--cluster_path '/home/ubuntu/code/hilander/hilander/data/selectbydensity_0.8_0.9_iter0.pkl' \
--finetune true --onehead true --loss smoothap --infrequent_eval 1
```

* `--trainset`, `testset`: $L_{in}$ training set, test set annotation txt file;
* `--cluster_path`: cluster data pkl file.

## Testing & Generate Features

### test

Test on the iNaturalist dataset, run `scripts/test_inat.sh`:

```commandline
CUDA_VISIBLE_DEVICES=0,1 python src/evaluate_model.py \
--dataset Inaturalist \
--bs 384 \
--source_path ~/code/Smooth_AP/data/ --embed_dim 128 \
--resume "INATURALIST_RESNET50_2021-9-2-4-21-21/checkpoint.pth.tar" \
--class_num 948 --loss smoothap \
--trainset lin_train_set1.txt \
--testset Inaturalist_test_set1.txt \
--linsize 29011 --uinsize 18403
```

* `--resume`: test model path;
* `--linsize`, `uinsize`: data size of $L_{in}$ and $U_{in}$.

### generate features

Generate features and labels for next step clustering, run `scripts/get-features-inat.sh`:

```commandline
CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 python src/get_features.py \
--dataset Inaturalist --lr 1e-5 --fc_lr_mul 1 \
--n_epochs 400 --bs 384 \
--source_path ~/code/Smooth_AP/data/ --embed_dim 128 \
--resume INATURALIST_RESNET50_2021-9-2-4-21-21/checkpoint.pth.tar \
--finetune false --onehead ture --get_features true --iter 1 \
--class_num 948 --loss smoothap \
--trainset lin_train_set1.txt \
--all_trainset train_set1.txt \
--testset test_set1.txt \
--linsize 29011 --uinsize 18403 \
--cluster_path ~/code/hilander/hilander/data/selectbydensity_0.8_0.9_iter0.pkl
```

* `--iter`: training iteration, start from iter 0;
* `--linsize`, `uinsize`: data size of $L_{in}$ and $U_{in}$, dose not change during iterations;
* `--cluster_path`: last iteration cluster data pkl path, effective only `iter>0`.

Note that while generating data, for all the training data set, data is following the order of $L_{in}$, $U$.

## ViT backbone

* For training in iteration 0, run `scripts/train_vit.sh`; 
* For training in iteration k (k>0), run `scripts/finetune_1head_vit.sh`; 
* For testing, run `scripts/test_vit.sh`;
* For generating features, run `scripts/get-features-vit.sh`.

## Paper

If you find this work useful, please consider citing:

```
@InProceedings{Brown20,
  author       = "Andrew Brown and Weidi Xie and Vicky Kalogeiton and Andrew Zisserman ",
  title        = "Smooth-AP: Smoothing the Path Towards Large-Scale Image Retrieval",
  booktitle    = "European Conference on Computer Vision (ECCV), 2020.",
  year         = "2020",
}
```