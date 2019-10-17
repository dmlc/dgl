# DGL - Knowledge Graph Embedding


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
 
- TransE
- DistMult
- ComplEx
- RESCAL

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

## Datasets

DGL-KE provides five knowledge graphs:

- [FB15k](https://s3.us-east-2.amazonaws.com/dgl.ai/dataset/{FB15k}.zip)
- [FB15k-237](https://s3.us-east-2.amazonaws.com/dgl.ai/dataset/{FB15k-237}.zip)
- [wn18](https://s3.us-east-2.amazonaws.com/dgl.ai/dataset/{wn18}.zip)
- [wn18rr](https://s3.us-east-2.amazonaws.com/dgl.ai/dataset/{wn18rr}.zip)
- [Freebase](https://s3.us-east-2.amazonaws.com/dgl.ai/dataset/{Freebase}.zip)

Users can specify one of the datasets with `--dataset` in `train.py` and `eval.py`.

## Performance

The speed is measured with 16 CPU cores and one Nvidia V100 GPU.

The speed on FB15k

|  Models | TransE | DistMult | ComplEx | RESCAL |
|---------|--------|----------|---------|--------|
|MAX_STEPS| 20000  | 100000   | 100000  | 30000  |
|TIME     | 411s   | 690s     | 806s    | 1800s  |

The accuracy on FB15k

|  Models  |  MR   |  MRR  | HITS@1 | HITS@3 | HITS@10 |
|----------|-------|-------|--------|--------|---------|
| TransE   | 69.12 | 0.656 | 0.567  | 0.718  | 0.802   |
| DistMult | 43.35 | 0.783 | 0.713  | 0.837  | 0.897   |
| ComplEx  | 51.99 | 0.785 | 0.720  | 0.832  | 0.889   |
| RESCAL   | 130.89| 0.668 | 0.597  | 0.720  | 0.800   | 

In comparison, GraphVite uses 4 GPUs and takes 14 minutes. Thus, DGL-KE trains TransE on FB15k twice as fast as GraphVite while using much few resources. More performance information on GraphVite can be found [here](https://github.com/DeepGraphLearning/graphvite).

The speed on wn18

|  Models | TransE | DistMult | ComplEx | RESCAL |
|---------|--------|----------|---------|--------|
|MAX_STEPS| 40000  | 10000    | 20000   | 20000  |
|TIME     | 719s   | 126s     | 266s    | 333s   |

The accuracy on wn18

|  Models  |  MR    |  MRR  | HITS@1 | HITS@3 | HITS@10 |
|----------|--------|-------|--------|--------|---------|
| TransE   | 321.35 | 0.760 | 0.652  | 0.850  | 0.940   |
| DistMult | 271.09 | 0.769 | 0.639  | 0.892  | 0.949   |
| ComplEx  | 276.37 | 0.935 | 0.916  | 0.950  | 0.960   |
| RESCAL   | 579.54 | 0.846 | 0.791  | 0.898  | 0.931   |

The speed on Freebase

|  Models | DistMult | ComplEx |
|---------|----------|---------|
|MAX_STEPS| 3200000  | 3200000 |
|TIME     | 2.44h    | 2.94h   |

The accuracy on Freebase (it is tested when 100,000 negative edges are sampled for each positive edge).

|  Models  |  MR    |  MRR  | HITS@1 | HITS@3 | HITS@10 |
|----------|--------|-------|--------|--------|---------|
| DistMul  | 6159.1 | 0.716 | 0.690  | 0.729  | 0.760   |
| ComplEx  | 6888.8 | 0.716 | 0.697  | 0.728  | 0.760   |

The configuration for reproducing the performance results can be found [here](https://github.com/dmlc/dgl/blob/master/apps/kg/config/best_config.sh).

## Usage

DGL-KE doesn't require installation. The package contains two scripts `train.py` and `eval.py`.

* `train.py` trains knowledge graph embeddings and outputs the trained node embeddings
and relation embeddings.

* `eval.py` reads the pre-trained node embeddings and relation embeddings and evaluate
how accurate to predict the tail node when given (head, rel, ?), and predict the head node
when given (?, rel, tail).

### Input formats:

DGL-KE supports two knowledge graph input formats. A knowledge graph is stored
using five files.

Format 1:

- entities.dict contains pairs of (entity Id, entity name). The number of rows is the number of entities (nodes).
- relations.dict contains pairs of (relation Id, relation name). The number of rows is the number of relations.
- train.txt stores edges in the training set. They are stored as triples of (head, rel, tail).
- valid.txt stores edges in the validation set. They are stored as triples of (head, rel, tail).
- test.txt stores edges in the test set. They are stored as triples of (head, rel, tail).

Format 2:

- entity2id.txt contains pairs of (entity name, entity Id). The number of rows is the number of entities (nodes).
- relation2id.txt contains pairs of (relation name, relation Id). The number of rows is the number of relations.
- train.txt stores edges in the training set. They are stored as triples of (head, tail, rel).
- valid.txt stores edges in the validation set. They are stored as a triple of (head, tail, rel).
- test.txt stores edges in the test set. They are stored as a triple of (head, tail, rel).

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
python3 train.py --model DistMult --dataset FB15k --batch_size 1024 \
    --neg_sample_size 256 --hidden_dim 2000 --gamma 500.0 --lr 0.1 --max_step 100000 \
    --batch_size_eval 16 --gpu 0 --valid --test -adv
```

Train KGE models with mixed CPUs and GPUs.

```bash
python3 train.py --model DistMult --dataset FB15k --batch_size 1024 \
    --neg_sample_size 256 --hidden_dim 2000 --gamma 500.0 --lr 0.1 --max_step 100000 \
    --batch_size_eval 16 --gpu 0 --valid --test -adv --mix_cpu_gpu
```

Train embeddings and verify it later.

```bash
python3 train.py --model DistMult --dataset FB15k --batch_size 1024 \
    --neg_sample_size 256 --hidden_dim 2000 --gamma 500.0 --lr 0.1 --max_step 100000 \
    --batch_size_eval 16 --gpu 0 --valid -adv --save_emb DistMult_FB15k_emb

python3 eval.py --model_name DistMult --dataset FB15k --hidden_dim 2000 \
    --gamma 500.0 --batch_size 16 --gpu 0 --model_path DistMult_FB15k_emb/

```

Train embeddings with multi-processing. This currently doesn't work in MXNet.
```bash
python3 train.py --model DistMult --dataset FB15k --batch_size 1024 \
    --neg_sample_size 256 --hidden_dim 2000 --gamma 500.0 --lr 0.07 --max_step 3000 \
    --batch_size_eval 16 --regularization_coef 0.000001 --valid --test -adv --num_proc 8
```
