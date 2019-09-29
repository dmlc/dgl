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
    --neg_sample_size 256 --hidden_dim 2000 --gamma 500.0 --adversarial_temperature 1.0 \
    --lr 0.1 --max_step 10000 --batch_size_eval 16 --regularization_coef 0.000002 --gpu 0 \
    --train --valid --test -adv --uni_weight --eval_interval 5000 \
    --opt RowAdagrad --neg_sample_size_valid 1000 --regularization_norm 3 --num_proc 1

Test average MR at [9999/10000]: 48.060655820961216
Test average MRR at [9999/10000]: 0.720459233368271
Test average HITS@1 at [9999/10000]: 0.6240879619441012
Test average HITS@3 at [9999/10000]: 0.7931895515566014
Test average HITS@10 at [9999/10000]: 0.8793655093023649

DGLBACKEND=pytorch python3 main.py --model ComplEx --dataset FB15k --batch_size 1024 \
    --neg_sample_size 256 --hidden_dim 2000 --gamma 500.0 --adversarial_temperature 1.0 \
    --lr 0.2 --max_step 20000 --batch_size_eval 16 --regularization_coef 0.000002 --gpu 1 \
    --train --valid --test -adv --uni_weight --eval_interval 2000 \
    --opt RowAdagrad --neg_sample_size_valid 1000 --log_interval 1000 --regularization_norm 3

Test average MR at [19999/20000]: 52.41770919740651
Test average MRR at [19999/20000]: 0.7520510504495108
Test average HITS@1 at [19999/20000]: 0.6723011291496673
Test average HITS@3 at [19999/20000]: 0.8112271673071388
Test average HITS@10 at [19999/20000]: 0.8815070000507863

DGLBACKEND=pytorch python3 main.py --model DistMult --dataset FB15k --batch_size 1024 \
    --neg_sample_size 256 --hidden_dim 400 --gamma 500.0 --adversarial_temperature 1.0 \
    --lr 0.2 --max_step 20000 --batch_size_eval 16 --regularization_coef 0.000002 --gpu 1 \
    --train --valid --test -adv --uni_weight --eval_interval 20000 \
    --opt RowAdagrad --neg_sample_size_valid 1000 --regularization_norm 3 --num_proc 1

Test average MR at [19999/20000]: 55.74326657750842
Test average MRR at [19999/20000]: 0.6739109335294652
Test average HITS@1 at [19999/20000]: 0.5618662287755413
Test average HITS@3 at [19999/20000]: 0.7559546985830611
Test average HITS@10 at [19999/20000]: 0.8616072184320563

DGLBACKEND=pytorch python3 main.py --model TransE --dataset FB15k --batch_size 1024 \
    --neg_sample_size 256 --hidden_dim 2000 --gamma 24.0 --adversarial_temperature 1.0 \
    --lr 0.01 --max_step 10000 --batch_size_eval 16 --regularization_coef 0.000002 --gpu 0 \
    --train --valid --test -adv --uni_weight --eval_interval 5000 --opt RowAdagrad \
    --neg_sample_size_valid 1000  --regularization_norm 3 --num_proc 1
Test average MR at [9999/10000]: 67.90176228606254
Test average MRR at [9999/10000]: 0.6450653824918221
Test average HITS@1 at [9999/10000]: 0.5480523437896768
Test average HITS@3 at [9999/10000]: 0.7109326065243521
Test average HITS@10 at [9999/10000]: 0.8047942306715647

DGLBACKEND=pytorch python3 main.py --model TransE --dataset FB15k --batch_size 1024 \
    --neg_sample_size 256 --hidden_dim 400 --gamma 24.0 --adversarial_temperature 1.0 \
    --lr 0.01 --max_step 10000 --batch_size_eval 16 --regularization_coef 0.000002 --gpu 0 \
    --train --valid --test -adv --uni_weight --eval_interval 5000 --opt RowAdagrad \
    --neg_sample_size_valid 1000  --regularization_norm 3 --num_proc 1
Test average MR at [9999/10000]: 97.97447986321545
Test average MRR at [9999/10000]: 0.5183488023519718
Test average HITS@1 at [9999/10000]: 0.36015980768905215
Test average HITS@3 at [9999/10000]: 0.6406697025613245
Test average HITS@10 at [9999/10000]: 0.7617781991163176

DGLBACKEND=pytorch python3 main.py --model ComplEx --dataset FB15k --batch_size 1024 \
    --neg_sample_size 256 --hidden_dim 400 --gamma 500.0 --adversarial_temperature 1.0 \
    --lr 0.2 --max_step 30000 --batch_size_eval 16 --train --test -adv --uni_weight \
    --eval_interval 2000 --opt RowAdagrad --neg_sample_size_valid 10000 \
    --gpu 2 --log_interval 1000 --regularization_coef 0.000001 --regularization_norm 3

Test average MR at [29999/30000]: 92.11875539604883
Test average MRR at [29999/30000]: 0.7668106483736371
Test average HITS@1 at [29999/30000]: 0.7040933791538996
Test average HITS@3 at [29999/30000]: 0.812699971221073
Test average HITS@10 at [29999/30000]: 0.870164717035432
```

Train with sparse embeddings with mixed CPUs and GPUs.
```bash
DGLBACKEND=pytorch python3 main.py --model DistMult --dataset FB15k --batch_size 1024 \
    --neg_sample_size 256 --hidden_dim 2000 --gamma 500.0 --adversarial_temperature 1.0 \
    --lr 0.1 --max_step 10000 --batch_size_eval 16 --regularization_coef 0.000002 --gpu 0 \
    --train --valid --test -adv --uni_weight --eval_interval 5000 \
    --opt RowAdagrad --neg_sample_size_valid 1000 --mix_cpu_gpu --regularization_norm 3 --num_proc 1

Test average MR at [9999/10000]: 47.51687799427807
Test average MRR at [9999/10000]: 0.7439476664953767
Test average HITS@1 at [9999/10000]: 0.6638367388397014
Test average HITS@3 at [9999/10000]: 0.8018316940630766
Test average HITS@10 at [9999/10000]: 0.8789422897868666

DGLBACKEND=pytorch python3 main.py --model TransE --dataset FB15k --batch_size 1024 \
     --neg_sample_size 256 --hidden_dim 2000 --gamma 24.0 --adversarial_temperature 1.0 \
     --lr 0.01 --max_step 10000 --batch_size_eval 16 --regularization_coef 0.000002 --gpu 0 \
     --train --valid --test -adv --uni_weight --eval_interval 5000 --opt RowAdagrad \
     --neg_sample_size_valid 1000 --mix_cpu_gpu  --regularization_norm 3 --num_proc 1

Test average HITS@10 at [9999/10000]: 0.7888727124985188
Test average MRR at [9999/10000]: 0.6022999231772103
Test average MR at [9999/10000]: 105.2224441773459
Test average HITS@3 at [9999/10000]: 0.6906434629513636
Test average HITS@1 at [9999/10000]: 0.484569416464932

```

Train embeddings and verify it later.
```bash
DGLBACKEND=pytorch python3 main.py --model DistMult --dataset FB15k --batch_size 1024 \
    --neg_sample_size 256 --hidden_dim 2000 --gamma 500.0 --adversarial_temperature 1.0 \
    --lr 0.001 --max_step 10000 --batch_size_eval 16 --regularization_coef 0.000002 --gpu 0 \
    --train --valid -adv --uni_weight --eval_interval 5000 --neg_sample_size_valid 1000 \
    --save_emb

DGLBACKEND=pytorch python3 eval.py --model_name DistMult --dataset FB15k --hidden_dim 2000 \
    --gamma 500.0 --batch_size 16 --gpu 0 --model_path ckpts/DistMult_FB15k_xx/
```

Train embeddings with multi-processing
```bash
DGLBACKEND=pytorch python3 main.py --model DistMult --dataset FB15k --batch_size 1024 \
    --neg_sample_size 256 --hidden_dim 2000 --gamma 500.0 --adversarial_temperature 1.0 --lr 0.07 \
    --max_step 3000 --batch_size_eval 16 --regularization_coef 0.000001 --train --valid --test -adv \
    --uni_weight --num_proc 8 --eval_interval 5000 --opt RowAdagrad --regularization_norm 3

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
Train with dense model weights.
```bash
DGLBACKEND=mxnet python3 main.py --model DistMult --dataset FB15k --batch_size 1024 \
     --neg_sample_size 256 --hidden_dim 2000 --gamma 500.0 --adversarial_temperature 1.0 \
      --lr 0.001 --max_step 10000 --batch_size_eval 16 --regularization_coef 0.0015 --gpu 0 \
      --train --valid --test -adv --uni_weight --eval_interval 5000 --neg_sample_size_valid 1000

Test average MR at [9999/10000]: 60.27311201774136
Test average HITS@3 at [9999/10000]: 0.808696314604459
Test average HITS@1 at [9999/10000]: 0.616156828223663
Test average MRR at [9999/10000]: 0.721805054329181
Test average HITS@10 at [9999/10000]: 0.8771309102605339

DGLBACKEND=mxnet python3 main.py --model TransE --dataset FB15k --batch_size 1024 \
     --neg_sample_size 256 --hidden_dim 2000 --gamma 24.0 --adversarial_temperature 1.0 \
     --lr 0.0002 --max_step 10000 --batch_size_eval 16 --gpu 0 --train --valid --test -adv --uni_weight \
     --eval_interval 5000 --neg_sample_size_valid 1000

Test average MR at [19999/20000]: 150.78540231247143
Test average HITS@3 at [19999/20000]: 0.6181374955561951
Test average HITS@10 at [19999/20000]: 0.7406595452929525
Test average HITS@1 at [19999/20000]: 0.37245856680943273
Test average MRR at [19999/20000]: 0.5127131081156031
```

Train with sparse model weights.
```bash
DGLBACKEND=mxnet  python3 main.py --model DistMult --dataset FB15k --batch_size 1024 \
      --neg_sample_size 256 --hidden_dim 2000 --gamma 500.0 --adversarial_temperature 1.0  \
      --lr 0.001 --max_step 10000 --batch_size_eval 16 --regularization_coef 0.0000001 --gpu 0 \
      --train --valid --test -adv --uni_weight --eval_interval 5000 --neg_sample_size_valid 1000

Test average HITS@10 at [19999/20000]: 0.8702324321579117
Test average HITS@1 at [19999/20000]: 0.5143386771850823
Test average MR at [19999/20000]: 68.11709637554807
Test average MRR at [19999/20000]: 0.6543895352041291
Test average HITS@3 at [19999/20000]: 0.7696077601530362

DGLBACKEND=mxnet  python3 main.py --model DistMult --dataset FB15k --batch_size 1024 \
    --neg_sample_size 256 --hidden_dim 400 --gamma 500.0 --adversarial_temperature 1.0 \
    --lr 0.002 --max_step 10000 --batch_size_eval 16 --regularization_coef 0.0000001 --gpu 0 \
    --train --valid --test -adv --uni_weight --eval_interval 5000 --neg_sample_size_valid 1000

Test average MRR at [9999/10000]: 0.5789892546035429
Test average HITS@1 at [9999/10000]: 0.42664759357383486
Test average HITS@10 at [9999/10000]: 0.8271825430414247
Test average MR at [9999/10000]: 72.41542381202282
Test average HITS@3 at [9999/10000]: 0.6962299605559411

DGLBACKEND=mxnet python3 main.py --model TransE --dataset FB15k --batch_size 1024 \
     --neg_sample_size 256 --hidden_dim 2000 --gamma 24.0 --adversarial_temperature 1.0 \
     --lr 0.0002 --max_step 10000 --batch_size_eval 16 --regularization_coef 0.00001  --gpu 0 \
     --train --valid --test -adv --uni_weight --eval_interval 5000 \
     --neg_sample_size_valid 1000 --log_interval 1000

Test average MRR at [9999/10000]: 0.6209599205127202
Test average HITS@1 at [9999/10000]: 0.5025145624492008
Test average MR at [9999/10000]: 68.5889748713086
Test average HITS@10 at [9999/10000]: 0.8101801679761582
Test average HITS@3 at [9999/10000]: 0.709081211053915

DGLBACKEND=mxnet python3 main.py --model TransE --dataset FB15k --batch_size 1024 \
     --neg_sample_size 256 --hidden_dim 2000 --gamma 24.0 --adversarial_temperature 1.0 \
     --lr 0.0002 --max_step 10000 --batch_size_eval 16 --regularization_coef 0.00001  --gpu 0 \
     --train --valid --test -adv --uni_weight --eval_interval 5000 \
     --neg_sample_size_valid 1000 --log_interval 1000

Test average MR at [9999/10000]: 145.42419292038394
Test average HITS@3 at [9999/10000]: 0.5686208122428942
Test average HITS@1 at [9999/10000]: 0.22887711398147992
Test average MRR at [9999/10000]: 0.41974998201318703
Test average HITS@10 at [9999/10000]: 0.7235445480862014
```

To test the model performance on Freebase, you can run the script `config/debug_fb.sh`:
Train embeddings on Freebase with multi-processing on X1.
```bash
DGLBACKEND=pytorch python3 main.py --model ComplEx --dataset Freebase --batch_size 1024 \
    --neg_sample_size 256 --hidden_dim 400 --gamma 500.0 --adversarial_temperature 1.0 \
    --lr 0.1 --max_step 50000 --batch_size_eval 128 --train --test -adv --uni_weight \
    --eval_interval 300000 --opt RowAdagrad \
    --neg_sample_size_test 10000 --eval_percent 0.2 --num_proc 64 --log_interval 10000 \
    --regularization_norm 3
Test average MR at [0/50000]: 754.5566055566055
Test average MRR at [0/50000]: 0.7333319016877765
Test average HITS@1 at [0/50000]: 0.7182952182952183
Test average HITS@3 at [0/50000]: 0.7409752409752409
Test average HITS@10 at [0/50000]: 0.7587412587412588
test time: 7678.401087999344
```

### MXNet

Train with sparse model weights.
```bash
DGLBACKEND=mxnet  python3 main.py --model DistMult --dataset FB15k --batch_size 1024 \
      --neg_sample_size 256 --hidden_dim 2000 --gamma 500.0 --adversarial_temperature 1.0  \
      --lr 0.001 --max_step 10000 --batch_size_eval 16 --regularization_coef 0.0000001 --gpu 0 \
      --train --valid --test -adv --uni_weight --eval_interval 5000 --neg_sample_size_valid 1000

Test average HITS@10 at [19999/20000]: 0.8702324321579117
Test average HITS@1 at [19999/20000]: 0.5143386771850823
Test average MR at [19999/20000]: 68.11709637554807
Test average MRR at [19999/20000]: 0.6543895352041291
Test average HITS@3 at [19999/20000]: 0.7696077601530362

DGLBACKEND=mxnet  python3 main.py --model DistMult --dataset FB15k --batch_size 1024 \
    --neg_sample_size 256 --hidden_dim 400 --gamma 500.0 --adversarial_temperature 1.0 \
    --lr 0.002 --max_step 10000 --batch_size_eval 16 --regularization_coef 0.0000001 --gpu 0 \
    --train --valid --test -adv --uni_weight --eval_interval 5000 --neg_sample_size_valid 1000

Test average MRR at [9999/10000]: 0.5789892546035429
Test average HITS@1 at [9999/10000]: 0.42664759357383486
Test average HITS@10 at [9999/10000]: 0.8271825430414247
Test average MR at [9999/10000]: 72.41542381202282
Test average HITS@3 at [9999/10000]: 0.6962299605559411

DGLBACKEND=mxnet python3 main.py --model TransE --dataset FB15k --batch_size 1024 \
     --neg_sample_size 256 --hidden_dim 2000 --gamma 24.0 --adversarial_temperature 1.0 \
     --lr 0.0002 --max_step 10000 --batch_size_eval 16 --regularization_coef 0.00001  --gpu 0 \
     --train --valid --test -adv --uni_weight --eval_interval 5000 \
     --neg_sample_size_valid 1000 --log_interval 1000

Test average HITS@1 at [9999/10000]: 0.3090348902168577
Test average MR at [9999/10000]: 126.96384858898614
Test average HITS@10 at [9999/10000]: 0.7800528177955341
Test average MRR at [9999/10000]: 0.49837393189051155
Test average HITS@3 at [9999/10000]: 0.6580301670870647

DGLBACKEND=mxnet python3 main.py --model TransE --dataset FB15k --batch_size 1024 \
     --neg_sample_size 256 --hidden_dim 2000 --gamma 24.0 --adversarial_temperature 1.0 \
     --lr 0.0002 --max_step 10000 --batch_size_eval 16 --regularization_coef 0.00001  --gpu 0 \
     --train --valid --test -adv --uni_weight --eval_interval 5000 \
     --neg_sample_size_valid 1000 --regularization_norm 3 --log_interval 1000

Test average MR at [9999/10000]: 145.42419292038394
Test average HITS@3 at [9999/10000]: 0.5686208122428942
Test average HITS@1 at [9999/10000]: 0.22887711398147992
Test average MRR at [9999/10000]: 0.41974998201318703
Test average HITS@10 at [9999/10000]: 0.7235445480862014
```
