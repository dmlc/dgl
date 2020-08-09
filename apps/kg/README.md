# DGL - Knowledge Graph Embedding

**Note: DGL-KE is moved to [here](https://github.com/awslabs/dgl-ke). DGL-KE in this folder is deprecated.**


## Introduction

DGL-KE is a DGL-based package for computing node embeddings and relation embeddings of
knowledge graphs efficiently. This package is adapted from
[KnowledgeGraphEmbedding](https://github.com/DeepGraphLearning/KnowledgeGraphEmbedding).
We enable fast and scalable training of knowledge graph embedding,
while still keeping the package as extensible as
[KnowledgeGraphEmbedding](https://github.com/DeepGraphLearning/KnowledgeGraphEmbedding).
On a single machine,
it takes only a few minutes for medium-size knowledge graphs, such as FB15k and wn18, and
takes a couple of hours on Freebase, which has hundreds of millions of edges.

DGL-KE includes the following knowledge graph embedding models:
 
- TransE (TransE_l1 with L1 distance and TransE_l2 with L2 distance)
- DistMult
- ComplEx
- RESCAL
- TransR
- RotatE

It will add other popular models in the future.

DGL-KE supports multiple training modes:

- CPU training
- GPU training
- Joint CPU & GPU training
- Multiprocessing training on CPUs

For joint CPU & GPU training, node embeddings are stored on CPU and mini-batches are trained on GPU. This is designed for training KGE models on large knowledge graphs

For multiprocessing training, each process train mini-batches independently and use shared memory for communication between processes. This is designed to train KGE models on large knowledge graphs with many CPU cores.

We will support multi-GPU training and distributed training in a near future.

## Requirements

The package can run with both Pytorch and MXNet. For Pytorch, it works with Pytorch v1.2 or newer.
For MXNet, it works with MXNet 1.5 or newer.

## Built-in Datasets

DGL-KE provides five built-in knowledge graphs:

| Dataset | #nodes | #edges | #relations |
|---------|--------|--------|------------|
| [FB15k](https://data.dgl.ai/dataset/FB15k.zip) | 14951 | 592213 | 1345 |
| [FB15k-237](https://data.dgl.ai/dataset/FB15k-237.zip) | 14541 | 310116 | 237 |
| [wn18](https://data.dgl.ai/dataset/wn18.zip) | 40943 | 151442 | 18 |
| [wn18rr](https://data.dgl.ai/dataset/wn18rr.zip) | 40943 | 93003 | 11 |
| [Freebase](https://data.dgl.ai/dataset/Freebase.zip) | 86054151 | 338586276 | 14824 |

Users can specify one of the datasets with `--dataset` in `train.py` and `eval.py`.

## Performance
The 1 GPU speed is measured with 8 CPU cores and one Nvidia V100 GPU. (AWS P3.2xlarge)
The 8 GPU speed is measured with 64 CPU cores and eight Nvidia V100 GPU. (AWS P3.16xlarge)

The speed on FB15k 1GPU

|  Models | TransE_l1 | TransE_l2 | DistMult | ComplEx | RESCAL | TransR | RotatE |
|---------|-----------|-----------|----------|---------|--------|--------|--------|
|MAX_STEPS| 48000     | 32000     | 40000    | 100000  | 32000  | 32000  | 20000  |
|TIME     | 370s      | 270s      | 312s     | 282s    | 2095s  | 1556s  | 1861s  |

The accuracy on FB15k

|  Models   |  MR   |  MRR  | HITS@1 | HITS@3 | HITS@10 |
|-----------|-------|-------|--------|--------|---------|
| TransE_l1 | 44.18 | 0.675 | 0.551  | 0.774  | 0.861   |
| TransE_l2 | 46.71 | 0.665 | 0.551  | 0.804  | 0.846   |
| DistMult  | 61.04 | 0.725 | 0.625  | 0.837  | 0.883   |
| ComplEx   | 64.59 | 0.785 | 0.718  | 0.835  | 0.889   |
| RESCAL    | 122.3 | 0.669 | 0.598  | 0.711  | 0.793   |
| TransR    | 59.86 | 0.676 | 0.591  | 0.735  | 0.814   |
| RotatE    | 43.66 | 0.728 | 0.632  | 0.801  | 0.874   |


The speed on FB15k 8GPU

|  Models | TransE_l1 | TransE_l2 | DistMult | ComplEx | RESCAL | TransR | RotatE |
|---------|-----------|-----------|----------|---------|--------|--------|--------|
|MAX_STEPS| 6000      | 4000      | 5000     | 4000    | 4000   | 4000   | 2500   |
|TIME     | 88.93s    | 62.99s    | 72.74s   | 68.37s  | 245.9s | 203.9s | 126.7s |

The accuracy on FB15k

|  Models   |  MR   |  MRR  | HITS@1 | HITS@3 | HITS@10 |
|-----------|-------|-------|--------|--------|---------|
| TransE_l1 | 44.25 | 0.672 | 0.547  | 0.774  | 0.860   |
| TransE_l2 | 46.13 | 0.658 | 0.539  | 0.748  | 0.845   |
| DistMult  | 61.72 | 0.723 | 0.626  | 0.798  | 0.881   |
| ComplEx   | 65.84 | 0.754 | 0.676  | 0.813  | 0.880   |
| RESCAL    | 135.6 | 0.652 | 0.580  | 0.693  | 0.779   |
| TransR    | 65.27 | 0.676 | 0.591  | 0.736  | 0.811   |
| RotatE    | 49.59 | 0.683 | 0.581  | 0.759  | 0.848   |

In comparison, GraphVite uses 4 GPUs and takes 14 minutes. Thus, DGL-KE trains TransE on FB15k 9.5X as fast as GraphVite with 8 GPUs. More performance information on GraphVite can be found [here](https://github.com/DeepGraphLearning/graphvite).

The speed on wn18 1GPU

|  Models | TransE_l1 | TransE_l2 | DistMult | ComplEx | RESCAL | TransR | RotatE |
|---------|-----------|-----------|----------|---------|--------|--------|--------|
|MAX_STEPS| 32000     | 32000     | 20000    | 20000   | 20000  | 30000  | 24000  |
|TIME     | 531.5s    | 406.6s    | 284.1s   | 282.3s  | 443.6s | 766.2s | 829.4s |

The accuracy on wn18

|  Models   |  MR   |  MRR  | HITS@1 | HITS@3 | HITS@10 |
|-----------|-------|-------|--------|--------|---------|
| TransE_l1 | 318.4 | 0.764 | 0.602  | 0.929  | 0.949   |
| TransE_l2 | 206.2 | 0.561 | 0.306  | 0.800  | 0.944   |
| DistMult  | 486.0 | 0.818 | 0.711  | 0.921  | 0.948   |
| ComplEx   | 268.6 | 0.933 | 0.916  | 0.949  | 0.961   |
| RESCAL    | 536.6 | 0.848 | 0.790  | 0.900  | 0.927   |
| TransR    | 452.4 | 0.620 | 0.461  | 0.758  | 0.856   |
| RotatE    | 487.9 | 0.944 | 0.940  | 0.947  | 0.952   |

The speed on wn18 8GPU

|  Models | TransE_l1 | TransE_l2 | DistMult | ComplEx | RESCAL | TransR | RotatE |
|---------|-----------|-----------|----------|---------|--------|--------|--------|
|MAX_STEPS| 4000      | 4000      | 2500     | 2500    | 2500   | 2500   | 3000   |
|TIME     | 119.3s    | 81.1s     | 76.0s    | 58.0s   | 594.1s | 1168s  | 139.8s |

The accuracy on wn18

|  Models   |  MR   |  MRR  | HITS@1 | HITS@3 | HITS@10 |
|-----------|-------|-------|--------|--------|---------|
| TransE_l1 | 360.3 | 0.745 | 0.562  | 0.930  | 0.951   |
| TransE_l2 | 193.8 | 0.557 | 0.301  | 0.799  | 0.942   |
| DistMult  | 499.9 | 0.807 | 0.692  | 0.917  | 0.945   |
| ComplEx   | 476.7 | 0.935 | 0.926  | 0.943  | 0.949   |
| RESCAL    | 618.8 | 0.848 | 0.791  | 0.897  | 0.927   |
| TransR    | 513.1 | 0.659 | 0.491  | 0.821  | 0.871   |
| RotatE    | 466.2 | 0.944 | 0.940  | 0.945  | 0.951   |


The speed on Freebase (8 GPU)

|  Models | TransE_l2 | DistMult | ComplEx | TransR | RotatE |
|---------|-----------|----------|---------|--------|--------|
|MAX_STEPS| 320000   | 300000   | 360000  | 300000 | 300000 |
|TIME     | 7908s     | 7425s    | 8946s   | 16816s | 12817s |

The accuracy on Freebase (it is tested when 1000 negative edges are sampled for each positive edge).

|  Models   |  MR    |  MRR  | HITS@1 | HITS@3 | HITS@10 |
|-----------|--------|-------|--------|--------|---------|
| TransE_l2 | 22.4   | 0.756 | 0.688  | 0.800  | 0.882   |
| DistMul   | 45.4   | 0.833 | 0.812  | 0.843  | 0.872   |
| ComplEx   | 48.0   | 0.830 | 0.812  | 0.838  | 0.864   |
| TransR    | 51.2   | 0.697 | 0.656  | 0.716  | 0.771   |
| RotatE    | 93.3   | 0.770 | 0.749  | 0.780  | 0.805   |

The speed on Freebase (48 CPU)
This measured with 48 CPU cores on an AWS r5dn.24xlarge

|  Models | TransE_l2 | DistMult | ComplEx |
|---------|-----------|----------|---------|
|MAX_STEPS| 50000     | 50000    | 50000   |
|TIME     | 7002s     | 6340s    | 8133s   |

The accuracy on Freebase (it is tested when 1000 negative edges are sampled for each positive edge).

|  Models   |  MR    |  MRR  | HITS@1 | HITS@3 | HITS@10 |
|-----------|--------|-------|--------|--------|---------|
| TransE_l2 | 30.8   | 0.814 | 0.764  | 0.848  | 0.902   |
| DistMul   | 45.1   | 0.834 | 0.815  | 0.843  | 0.871   |
| ComplEx   | 44.9   | 0.837 | 0.819  | 0.845  | 0.870   |

The configuration for reproducing the performance results can be found [here](https://github.com/dmlc/dgl/blob/master/apps/kg/config/best_config.sh).

## Usage

DGL-KE doesn't require installation. The package contains two scripts `train.py` and `eval.py`.

* `train.py` trains knowledge graph embeddings and outputs the trained node embeddings
and relation embeddings.

* `eval.py` reads the pre-trained node embeddings and relation embeddings and evaluate
how accurate to predict the tail node when given (head, rel, ?), and predict the head node
when given (?, rel, tail).

### Input formats:

DGL-KE supports two knowledge graph input formats for user defined dataset

- raw_udd_[h|r|t], raw user defined dataset. In this format, user only need to provide triples and let the dataloader generate and manipulate the id mapping. The dataloader will generate two files: entities.tsv for entity id mapping and relations.tsv for relation id mapping. The order of head, relation and tail entities are described in [h|r|t], for example, raw_udd_trh means the triples are stored in the order of tail, relation and head. It should contains three files:
  - *train* stores the triples in the training set. In format of a triple, e.g., [src_name, rel_name, dst_name] and should follow the order specified in [h|r|t]
  - *valid* stores the triples in the validation set. In format of a triple, e.g., [src_name, rel_name, dst_name] and should follow the order specified in [h|r|t]
  - *test* stores the triples in the test set. In format of a triple, e.g., [src_name, rel_name, dst_name] and should follow the order specified in [h|r|t]

Format 2:
- udd_[h|r|t], user defined dataset. In this format, user should provide the id mapping for entities and relations. The order of head, relation and tail entities are described in [h|r|t], for example, raw_udd_trh means the triples are stored in the order of tail, relation and head. It should contains five files:
  - *entities* stores the mapping between entity name and entity Id
  - *relations* stores the mapping between relation name relation Id
  - *train* stores the triples in the training set. In format of a triple, e.g., [src_id, rel_id, dst_id] and should follow the order specified in [h|r|t]
  - *valid* stores the triples in the validation set. In format of a triple, e.g., [src_id, rel_id, dst_id] and should follow the order specified in [h|r|t]
  - *test* stores the triples in the test set. In format of a triple, e.g., [src_id, rel_id, dst_id] and should follow the order specified in [h|r|t]

### Output formats:

To save the trained embeddings, users have to provide the path with `--save_emb` when running
`train.py`. The saved embeddings are stored as numpy ndarrays.

* The node embedding is saved as `XXX_YYY_entity.npy`.

* The relation embedding is saved as `XXX_YYY_relation.npy`.

`XXX` is the dataset name and `YYY` is the model name.

### Command line parameters

Here are some examples of using the training script.

Train KGE models with GPU.

```bash
python3 train.py --model DistMult --dataset FB15k --batch_size 1024 --neg_sample_size 256 \
    --hidden_dim 400 --gamma 143.0 --lr 0.08 --batch_size_eval 16 --valid --test -adv \
    --gpu 0 --max_step 40000
```

Train KGE models with mixed multiple GPUs.

```bash
python3 train.py --model DistMult --dataset FB15k --batch_size 1024 --neg_sample_size 256 \
    --hidden_dim 400 --gamma 143.0 --lr 0.08 --batch_size_eval 16 --valid --test -adv \
    --max_step 5000 --mix_cpu_gpu --num_proc 8 --gpu 0 1 2 3 4 5 6 7 --async_update \
    --soft_rel_part --force_sync_interval 1000
```

Train embeddings and verify it later.

```bash
python3 train.py --model DistMult --dataset FB15k --batch_size 1024 --neg_sample_size 256 \
    --hidden_dim 400 --gamma 143.0 --lr 0.08 --batch_size_eval 16 --valid --test -adv \
     --gpu 0 --max_step 40000 --save_emb DistMult_FB15k_emb

python3 eval.py --model_name DistMult --dataset FB15k --hidden_dim 400 \
    --gamma 143.0 --batch_size 16 --gpu 0 --model_path DistMult_FB15k_emb/

```

Train embeddings with multi-processing. This currently doesn't work in MXNet.
```bash
python3 train.py --model TransE_l2 --dataset Freebase --batch_size 1000 \
    --neg_sample_size 200 --hidden_dim 400 --gamma 10 --lr 0.1 --max_step 50000 \
    --log_interval 100 --batch_size_eval 1000 --neg_sample_size_eval 1000 --test \
   -adv --regularization_coef 1e-9 --num_thread 1 --num_proc 48
```
