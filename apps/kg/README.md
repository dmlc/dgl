# DGL - Knowledge Graph Embedding


## Models

DGL-KE now support knowledge graph embedding models including:

- [x] TransE
- [x] TransH
- [x] TransR
- [x] TransD
- [x] RESCAL
- [x] DistMult
- [x] ComplEx
- [x] RotatE

## Datasets

DGL-KE is tested on datasets including:

- [x] FB15k
- [x] FB15k-237
- [x] wn18
- [x] wn18rr
- [ ] Freebase

## Usage

To reproduce reported results, you can run the model using commands in `config/best_config_dgl.sh`. Here is an example:

Train with sparse embeddings.
```bash
DGLBACKEND=pytorch python3 main.py --model DistMult --dataset FB15k --batch_size 1024 \
    --neg_sample_size 256 --hidden_dim 2000 --gamma 500.0 --lr 0.1 --max_step 10000 \
    --batch_size_eval 16 --gpu 0 --train --valid --test -adv

Test average MR at [9999/10000]: 48.060655820961216
Test average MRR at [9999/10000]: 0.720459233368271
Test average HITS@1 at [9999/10000]: 0.6240879619441012
Test average HITS@3 at [9999/10000]: 0.7931895515566014
Test average HITS@10 at [9999/10000]: 0.8793655093023649


DGLBACKEND=pytorch python3 main.py --model ComplEx --dataset FB15k --batch_size 1024 \
    --neg_sample_size 256 --hidden_dim 2000 --gamma 500.0 --lr 0.2 --max_step 10000 \
    --batch_size_eval 16 --gpu 0 --train --valid --test -adv

Test average HITS@10 at [9999/10000]: 0.85456177187754
Test average HITS@3 at [9999/10000]: 0.7738248442156597
Test average MRR at [9999/10000]: 0.715834589219438
Test average MR at [9999/10000]: 87.72186568680574
Test average HITS@1 at [9999/10000]: 0.6333480086697372


DGLBACKEND=pytorch python3 main.py --model TransE --dataset FB15k --batch_size 1024 \
    --neg_sample_size 256 --hidden_dim 2000 --gamma 24.0 --lr 0.01 --max_step 10000 \
    --batch_size_eval 16 --gpu 0 --train --valid --test -adv

Test average MR at [9999/10000]: 67.90176228606254
Test average MRR at [9999/10000]: 0.6450653824918221
Test average HITS@1 at [9999/10000]: 0.5480523437896768
Test average HITS@3 at [9999/10000]: 0.7109326065243521
Test average HITS@10 at [9999/10000]: 0.8047942306715647


DGLBACKEND=pytorch python3 main.py --model DistMult --dataset FB15k --batch_size 1024 \
    --neg_sample_size 256 --hidden_dim 400 --gamma 500.0 --lr 0.2 --max_step 10000 \
    --batch_size_eval 16 --gpu 0 --train --valid --test -adv

Test average MR at [9999/10000]: 77.8360285153075
Test average HITS@3 at [9999/10000]: 0.7113248442156597
Test average MRR at [9999/10000]: 0.6317390642641796
Test average HITS@1 at [9999/10000]: 0.5169838797073963
Test average HITS@10 at [9999/10000]: 0.8303220671904633


DGLBACKEND=pytorch python3 main.py --model ComplEx --dataset FB15k --batch_size 1024 \
    --neg_sample_size 256 --hidden_dim 400 --gamma 500.0 --lr 0.2 --max_step 10000 \
    --batch_size_eval 16 --train --test -adv --gpu 0 --regularization_coef 0.000001

Test average HITS@3 at [9999/10000]: 0.7175985505283121
Test average MRR at [9999/10000]: 0.6595120996930349
Test average MR at [9999/10000]: 148.70467014359252
Test average HITS@1 at [9999/10000]: 0.5713644676239501
Test average HITS@10 at [9999/10000]: 0.8155818206448117

DGLBACKEND=pytorch python3 main.py --model TransE --dataset FB15k --batch_size 1024 \
    --neg_sample_size 256 --hidden_dim 400 --gamma 24.0 --lr 0.01 --max_step 10000 \
    --batch_size_eval 16 --gpu 0 --train --valid --test -adv

Test average HITS@1 at [9999/10000]: 0.3604460173394744
Test average HITS@10 at [9999/10000]: 0.759685722026551
Test average MRR at [9999/10000]: 0.5171090122873108
Test average MR at [9999/10000]: 96.8555523570848
Test average HITS@3 at [9999/10000]: 0.6376236114874018

```

Train with sparse embeddings with mixed CPUs and GPUs.
```bash
DGLBACKEND=pytorch python3 main.py --model DistMult --dataset FB15k --batch_size 1024 \
    --neg_sample_size 256 --hidden_dim 2000 --gamma 500.0 --lr 0.1 --max_step 10000 \
    --batch_size_eval 16 --gpu 0 --train --valid --test -adv --mix_cpu_gpu

Test average MR at [9999/10000]: 47.51687799427807
Test average MRR at [9999/10000]: 0.7439476664953767
Test average HITS@1 at [9999/10000]: 0.6638367388397014
Test average HITS@3 at [9999/10000]: 0.8018316940630766
Test average HITS@10 at [9999/10000]: 0.8789422897868666

DGLBACKEND=pytorch python3 main.py --model ComplEx --dataset FB15k --batch_size 1024 \
    --neg_sample_size 256 --hidden_dim 2000 --gamma 500.0 --lr 0.2 --max_step 10000 \
    --batch_size_eval 16 --gpu 0 --train --valid --test -adv --mix_cpu_gpu

Test average HITS@1 at [9999/10000]: 0.5751574776483338
Test average MRR at [9999/10000]: 0.6809614106013072
Test average MR at [9999/10000]: 93.60952316445407
Test average HITS@10 at [9999/10000]: 0.8503539013817394
Test average HITS@3 at [9999/10000]: 0.7622002844757518

DGLBACKEND=pytorch python3 main.py --model TransE --dataset FB15k --batch_size 1024 \
    --neg_sample_size 256 --hidden_dim 2000 --gamma 24.0 --lr 0.01 --max_step 10000 \
    --batch_size_eval 16 --gpu 0 --train --valid --test -adv --mix_cpu_gpu

Test average HITS@1 at [9999/10000]: 0.5516628285017611
Test average HITS@3 at [9999/10000]: 0.7171413573557301
Test average MRR at [9999/10000]: 0.6488753519183479
Test average HITS@10 at [9999/10000]: 0.8089017204009753
Test average MR at [9999/10000]: 64.82082260904903
```

Train embeddings and verify it later.
```bash
DGLBACKEND=pytorch python3 main.py --model DistMult --dataset FB15k --batch_size 1024 \
    --neg_sample_size 256 --hidden_dim 2000 --gamma 500.0 --lr 0.1 --max_step 10000 \
    --batch_size_eval 16 --gpu 0 --train --valid -adv --save_emb

DGLBACKEND=pytorch python3 eval.py --model_name DistMult --dataset FB15k --hidden_dim 2000 \
    --gamma 500.0 --batch_size 16 --gpu 0 --model_path ckpts/DistMult_FB15k_xx/


DGLBACKEND=pytorch python3 main.py --model ComplEx --dataset FB15k --batch_size 1024 \
    --neg_sample_size 256 --hidden_dim 2000 --gamma 500.0 --lr 0.2 --max_step 10000 \
    --batch_size_eval 16 --gpu 0 --train --valid -adv --save_emb

DGLBACKEND=pytorch python3 eval.py --model_name ComplEx --dataset FB15k --hidden_dim 2000 \
    --gamma 500.0 --batch_size 16 --gpu 0 --model_path ckpts/ComplEx_FB15k_xx/


DGLBACKEND=pytorch python3 main.py --model TransE --dataset FB15k --batch_size 1024 \
    --neg_sample_size 256 --hidden_dim 2000 --gamma 24.0 --lr 0.01 --max_step 10000 \
    --batch_size_eval 16 --gpu 0 --train --valid -adv --save_emb

DGLBACKEND=pytorch python3 eval.py --model_name TransE --dataset FB15k --hidden_dim 2000 \
    --gamma 24.0 --batch_size 16 --gpu 0 --model_path ckpts/TransE_FB15k_xx/
```

Train embeddings with multi-processing
```bash
DGLBACKEND=pytorch python3 main.py --model DistMult --dataset FB15k --batch_size 1024 \
    --neg_sample_size 256 --hidden_dim 2000 --gamma 500.0 --lr 0.07 --max_step 3000 \
    --batch_size_eval 16 --regularization_coef 0.000001 --train --valid --test -adv --num_proc 8

Test average MR at [0/3000]: 62.4907908992416
Test average MRR at [0/3000]: 0.75353530039047
Test average HITS@1 at [0/3000]: 0.6787648970747562
Test average HITS@3 at [0/3000]: 0.8082340195016251
Test average HITS@10 at [0/3000]: 0.875406283856988

Test average MR at [0/3000]: 45.05694143167028
Test average MRR at [0/3000]: 0.7632975771949468
Test average HITS@1 at [0/3000]: 0.6876355748373102
Test average HITS@3 at [0/3000]: 0.8172451193058569
Test average HITS@10 at [0/3000]: 0.8899132321041214

Test average MR at [0/3000]: 45.66955579631636
Test average MRR at [0/3000]: 0.7443389093675862
Test average HITS@1 at [0/3000]: 0.6706392199349945
Test average HITS@3 at [0/3000]: 0.7903575297941495
Test average HITS@10 at [0/3000]: 0.8775731310942578

Test average MR at [0/3000]: 67.80335861321777
Test average MRR at [0/3000]: 0.756062329815256
Test average HITS@1 at [0/3000]: 0.685807150595883
Test average HITS@3 at [0/3000]: 0.804442036836403
Test average HITS@10 at [0/3000]: 0.8743228602383532

Test average MR at [0/3000]: 66.75297941495124
Test average MRR at [0/3000]: 0.7379322316142427
Test average HITS@1 at [0/3000]: 0.6543878656554712
Test average HITS@3 at [0/3000]: 0.800650054171181
Test average HITS@10 at [0/3000]: 0.8781148429035753

Test average MR at [0/3000]: 72.60834236186349
Test average MRR at [0/3000]: 0.755075951436731
Test average HITS@1 at [0/3000]: 0.6841820151679306
Test average HITS@3 at [0/3000]: 0.804442036836403
Test average HITS@10 at [0/3000]: 0.8721560130010835

Test average MR at [0/3000]: 49.08179848320693
Test average MRR at [0/3000]: 0.7505612896477916
Test average HITS@1 at [0/3000]: 0.6814734561213435
Test average HITS@3 at [0/3000]: 0.7990249187432286
Test average HITS@10 at [0/3000]: 0.8726977248104009

Test average MR at [0/3000]: 30.860780065005418
Test average MRR at [0/3000]: 0.7453079145378247
Test average HITS@1 at [0/3000]: 0.6657638136511376
Test average HITS@3 at [0/3000]: 0.7984832069339112
Test average HITS@10 at [0/3000]: 0.8813651137594799

```


### MXNet
Train with sparse embeddings.
```bash
DGLBACKEND=mxnet python3 main.py --model DistMult --dataset FB15k --batch_size 1024 \
    --neg_sample_size 256 --hidden_dim 2000 --gamma 500.0 --lr 0.1 --max_step 10000 \
    --batch_size_eval 16 --gpu 0 --train --valid --test -adv

Test average MRR at [9999/10000]: 0.7146810716281126
Test average HITS@10 at [9999/10000]: 0.8762953806556488
Test average MR at [9999/10000]: 49.36298597940937
Test average HITS@3 at [9999/10000]: 0.7883872934164183
Test average HITS@1 at [9999/10000]: 0.6161355323760499


DGLBACKEND=mxnet python3 main.py --model TransE --dataset FB15k --batch_size 1024 \
    --neg_sample_size 256 --hidden_dim 2000 --gamma 24.0 --lr 0.01 --max_step 10000 \
    --batch_size_eval 16 --gpu 0 --train --valid --test -adv

Test average HITS@3 at [9999/10000]: 0.7070238417772962
Test average HITS@1 at [9999/10000]: 0.5244429016526686
Test average MR at [9999/10000]: 67.70047073963696
Test average HITS@10 at [9999/10000]: 0.8050748442156597
Test average MRR at [9999/10000]: 0.6311273723585298


DGLBACKEND=mxnet python3 main.py --model DistMult --dataset FB15k --batch_size 1024 \
    --neg_sample_size 256 --hidden_dim 400 --gamma 500.0 --lr 0.2 --max_step 10000 \
    --batch_size_eval 16 --gpu 0 --train --valid --test -adv

Test average MRR at [9999/10000]: 0.5972367796993037
Test average HITS@1 at [9999/10000]: 0.479722636141967
Test average MR at [9999/10000]: 82.75723889189922
Test average HITS@3 at [9999/10000]: 0.6754013140070442
Test average HITS@10 at [9999/10000]: 0.8039487943646708


DGLBACKEND=mxnet python3 main.py --model TransE --dataset FB15k --batch_size 1024 \
    --neg_sample_size 256 --hidden_dim 400 --gamma 24.0 --lr 0.01 --max_step 10000 \
    --batch_size_eval 16 --gpu 0 --train --valid --test -adv

Test average MRR at [9999/10000]: 0.43204399210506417
Test average HITS@1 at [9999/10000]: 0.24382789217014358
Test average HITS@10 at [9999/10000]: 0.7209259008398808
Test average MR at [9999/10000]: 117.35780445678678
Test average HITS@3 at [9999/10000]: 0.5790944188566785
```

Train with sparse embeddings with mixed CPUs and GPUs.
```bash
DGLBACKEND=mxnet python3 main.py --model DistMult --dataset FB15k --batch_size 1024 \
    --neg_sample_size 256 --hidden_dim 2000 --gamma 500.0 --lr 0.1 --max_step 10000 \
    --batch_size_eval 16 --gpu 0 --train --valid --test -adv --mix_cpu_gpu

Test average HITS@3 at [9999/10000]: 0.7857457328637226
Test average MRR at [9999/10000]: 0.7118503745011753
Test average HITS@10 at [9999/10000]: 0.875575724735844
Test average HITS@1 at [9999/10000]: 0.6129267136277432
Test average MR at [9999/10000]: 47.41050020319697


DGLBACKEND=mxnet python3 main.py --model TransE --dataset FB15k --batch_size 1024 \
    --neg_sample_size 256 --hidden_dim 2000 --gamma 24.0 --lr 0.01 --max_step 10000 \
    --batch_size_eval 16 --gpu 0 --train --valid --test -adv --mix_cpu_gpu

Test average HITS@1 at [9999/10000]: 0.5229781901923598
Test average MR at [9999/10000]: 68.7329399214305
Test average HITS@3 at [9999/10000]: 0.7054575318341912
Test average MRR at [9999/10000]: 0.629640060618377
Test average HITS@10 at [9999/10000]: 0.8033561365483609
```

Train embeddings and verify it later.
```bash
DGLBACKEND=mxnet python3 main.py --model DistMult --dataset FB15k --batch_size 1024 \
    --neg_sample_size 256 --hidden_dim 2000 --gamma 500.0 --lr 0.1 --max_step 10000 \
    --batch_size_eval 16 --gpu 0 --train --valid -adv --save_emb

DGLBACKEND=mxnet python3 eval.py --model_name DistMult --dataset FB15k --hidden_dim 2000 \
    --gamma 500.0 --batch_size 16 --gpu 0 --model_path ckpts/DistMult_FB15k_xx/


DGLBACKEND=mxnet python3 main.py --model TransE --dataset FB15k --batch_size 1024 \
    --neg_sample_size 256 --hidden_dim 2000 --gamma 24.0 --lr 0.01 --max_step 10000 \
    --batch_size_eval 16 --gpu 0 --train --valid -adv --save_emb

DGLBACKEND=mxnet python3 eval.py --model_name TransE --dataset FB15k --hidden_dim 2000 \
    --gamma 24.0 --batch_size 16 --gpu 0 --model_path ckpts/TransE_FB15k_xx/
```

## Freebase
To test the model performance on Freebase, you can run the script `config/debug_fb.sh`:
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
