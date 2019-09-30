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

## Usage

## Examples

Train KGE models with GPU.

```bash
python3 main.py --model DistMult --dataset FB15k --batch_size 1024 \
    --neg_sample_size 256 --hidden_dim 2000 --gamma 500.0 --lr 0.1 --max_step 100000 \
    --batch_size_eval 16 --gpu 0 --train --valid --test -adv
```

Train KGE models with mixed CPUs and GPUs.

python3 main.py --model DistMult --dataset FB15k --batch_size 1024 \
    --neg_sample_size 256 --hidden_dim 2000 --gamma 500.0 --lr 0.1 --max_step 100000 \
    --batch_size_eval 16 --gpu 0 --train --valid --test -adv --mix_cpu_gpu

Train embeddings and verify it later.
```bash
python3 main.py --model DistMult --dataset FB15k --batch_size 1024 \
    --neg_sample_size 256 --hidden_dim 2000 --gamma 500.0 --lr 0.1 --max_step 100000 \
    --batch_size_eval 16 --gpu 0 --train --valid -adv --save_emb

python3 eval.py --model_name DistMult --dataset FB15k --hidden_dim 2000 \
    --gamma 500.0 --batch_size 16 --gpu 0 --model_path ckpts/DistMult_FB15k_xx/

```

Train embeddings with multi-processing. This currently doesn't work in MXNet.
```bash
python3 main.py --model DistMult --dataset FB15k --batch_size 1024 \
    --neg_sample_size 256 --hidden_dim 2000 --gamma 500.0 --lr 0.07 --max_step 3000 \
    --batch_size_eval 16 --regularization_coef 0.000001 --train --valid --test -adv --num_proc 8
```

## Freebase
Train embeddings on Freebase with multi-processing on X1.
```bash
DGLBACKEND=pytorch python3 main.py --model ComplEx --dataset Freebase --batch_size 1024 \
    --neg_sample_size 256 --hidden_dim 400 --gamma 500.0 \
    --lr 0.1 --max_step 50000 --batch_size_eval 128 --train --test -adv --eval_interval 300000 \
    --neg_sample_size_test 10000 --eval_percent 0.2 --num_proc 64
Test average MR at [0/50000]: 754.5566055566055
Test average MRR at [0/50000]: 0.7333319016877765
Test average HITS@1 at [0/50000]: 0.7182952182952183
Test average HITS@3 at [0/50000]: 0.7409752409752409
Test average HITS@10 at [0/50000]: 0.7587412587412588
test time: 7678.401087999344
```
