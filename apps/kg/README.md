# DGL - Knowledge Graph Embedding


## Models


DGL-KE now supports knowledge graph embedding models including:
 
- [x] TransE
- [x] DistMult
- [x] ComplEx

More models will be supported in a near future.

## Datasets

DGL-KE is tested on datasets including:

- [x] FB15k
- [x] FB15k-237
- [x] wn18
- [x] wn18rr
- [x] Freebase

## Performance

The speed is measured on an EC2 P3 instance on a Nvidia V100 GPU.

The speed on FB15k
|  Models | TrasnE | DistMult | ComplEx |
|---------|--------|----------|---------|
|MAX_STEPS| 20000  | 100000   | 100000  |
|TIME     | 411s   | 690s     | 806s    |

The accuracy on FB15k
|  Models  |  MR   |  MRR  | HITS@1 | HITS@3 | HITS@10 |
|----------|-------|-------|--------|--------|---------|
| TransE   | 69.12 | 0.656 | 0.567  | 0.718  | 0.802   |
| DistMult | 43.35 | 0.783 | 0.713  | 0.837  | 0.897   |
| ComplEx  | 51.99 | 0.785 | 0.720  | 0.832  | 0.889   |

The speed on wn18
|  Models | TrasnE | DistMult | ComplEx |
|---------|--------|----------|---------|
|MAX_STEPS| 40000  | 10000    | 20000   |
|TIME     | 719s   | 126s     | 266s    |

The accuracy on wn18
|  Models  |  MR    |  MRR  | HITS@1 | HITS@3 | HITS@10 |
|----------|--------|-------|--------|--------|---------|
| TransE   | 321.35 | 0.760 | 0.652  | 0.850  | 0.940   |
| DistMult | 271.09 | 0.769 | 0.639  | 0.892  | 0.949   |
| ComplEx  | 276.37 | 0.935 | 0.916  | 0.950  | 0.960   |

## Usage

The package supports two data formats for a knowledge graph.

Format 1:
    * entities.dict maps entity Id to entity name.
    * relations.dict maps relation Id to relation name.
    * train.txt stores the triples (head, rel, tail) in the training set.
    * valid.txt stores the triples (head, rel, tail) in the validation set.
    * test.txt stores the triples (head, rel, tail) in the test set.

Format 2:
    * entity2id.txt maps entity name to entity Id.
    * relation2id.txt maps relation name to relation Id.
    * train.txt stores the triples (head, tail, rel) in the training set.
    * valid.txt stores the triples (head, tail, rel) in the validation set.
    * test.txt stores the triples (head, tail, rel) in the test set.

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
    --batch_size_eval 16 --gpu 0 --valid -adv --save_emb

python3 eval.py --model_name DistMult --dataset FB15k --hidden_dim 2000 \
    --gamma 500.0 --batch_size 16 --gpu 0 --model_path ckpts/DistMult_FB15k_xx/

```

Train embeddings with multi-processing. This currently doesn't work in MXNet.
```bash
python3 train.py --model DistMult --dataset FB15k --batch_size 1024 \
    --neg_sample_size 256 --hidden_dim 2000 --gamma 500.0 --lr 0.07 --max_step 3000 \
    --batch_size_eval 16 --regularization_coef 0.000001 --valid --test -adv --num_proc 8
```

## Freebase
Train embeddings on Freebase with multi-processing on X1.
```bash
DGLBACKEND=pytorch python3 train.py --model ComplEx --dataset Freebase --batch_size 1024 \
    --neg_sample_size 256 --hidden_dim 400 --gamma 500.0 \
    --lr 0.1 --max_step 50000 --batch_size_eval 128 --test -adv --eval_interval 300000 \
    --neg_sample_size_test 10000 --eval_percent 0.2 --num_proc 64

Test average MR at [0/50000]: 754.5566055566055
Test average MRR at [0/50000]: 0.7333319016877765
Test average HITS@1 at [0/50000]: 0.7182952182952183
Test average HITS@3 at [0/50000]: 0.7409752409752409
Test average HITS@10 at [0/50000]: 0.7587412587412588
```
