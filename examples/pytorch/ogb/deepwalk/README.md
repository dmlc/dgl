# DeepWalk Example

- Paper link: [here](https://arxiv.org/pdf/1403.6652.pdf)
- Other implementation: [gensim](https://github.com/phanein/deepwalk), [deepwalk-c](https://github.com/xgfs/deepwalk-c)

The implementation includes multi-processing training with CPU and mixed training with CPU and multi-GPU.

## Dependencies
- PyTorch 1.5.0+

## Tested version
- PyTorch 1.5.0
- DGL 0.5.0

## Input data
Currently, we support two builtin dataset: youtube and blog. Use --data\_file youtube to select youtube dataset and --data\_file blog to select blog dataset.
The data is avaliable at  https://data.dgl.ai/dataset/DeepWalk/youtube.zip and https://data.dgl.ai/dataset/DeepWalk/blog.zip
The youtube.zip includes both youtube-net.txt, youtube-vocab.txt and youtube-label.txt; The blog.zip includes both blog-net.txt, blog-vocab.txt and blog-label.txt. 

For other datasets please pass the full path to the trainer through --data\_file and the format of a network file should follow:
```
1(node id) 2(node id)
1 3
1 4
2 4
...
```

### How to run the code
To run the code:
```
python3 deepwalk.py --data_file youtube --output_emb_file emb.txt --mix --lr 0.2 --gpus 0 1 2 3 --batch_size 100 --negative 5
```

### How to save the embedding
By default the trained embedding is saved under --output\_embe\_file FILE\_NAME as a numpy object.
To save the trained embedding in raw format(txt format), please use --save\_in\_txt argument.

### Evaluation

To evalutate embedding on multi-label classification, please refer to [here](https://github.com/ShawXh/Evaluate-Embedding)

YouTube (1M nodes).

| Implementation | Macro-F1 (%) <br> 1% &emsp;&emsp; 3% &emsp;&emsp; 5% &emsp;&emsp; 7% &emsp;&emsp; 9% | Micro-F1 (%) <br> 1% &emsp;&emsp; 3% &emsp;&emsp; 5% &emsp;&emsp; 7% &emsp;&emsp; 9% |
|----|----|----|
| gensim.word2vec(hs) | 28.73 &emsp; 32.51 &emsp; 33.67 &emsp; 34.28 &emsp; 34.79 | 35.73 &emsp; 38.34 &emsp; 39.37 &emsp; 40.08 &emsp; 40.77 | 
| gensim.word2vec(ns) | 28.18 &emsp; 32.25 &emsp; 33.56 &emsp; 34.60 &emsp; 35.22 | 35.35 &emsp; 37.69 &emsp; 38.08 &emsp; 40.24 &emsp; 41.09 | 
|        ours         | 24.58 &emsp; 31.23 &emsp; 33.97 &emsp; 35.41 &emsp; 36.48 | 38.93 &emsp; 43.17 &emsp; 44.73 &emsp; 45.42 &emsp; 45.92 | 

The comparison between running time is shown as below, where the numbers in the brackets denote time used on random-walk.

| Implementation | gensim.word2vec(hs) | gensim.word2vec(ns) | Ours |
|----|----|----|----|
| Time (s) |     27119.6(1759.8)    |    10580.3(1704.3)    | 428.89 |

Parameters.
- walk_length = 80, number_walks = 10, window_size = 5
- Ours: 4GPU (Tesla V100), lr = 0.2, batchs_size = 128, neg_weight = 5, negative = 1, num_thread = 4
- Others: workers = 8, negative = 5

Speeding-up with mixed CPU & multi-GPU. The used parameters are the same as above.
|  #GPUs   |   1   |   2   |   4   |
|----------|-------|-------|-------|
| Time (s) |1419.64| 952.04|428.89 |

## OGB Dataset
### How to load ogb data
You can run the code directly with:
```
python3 deepwalk --ogbl_name xxx --load_from_ogbl
```
However, ogb.linkproppred might not be compatible with mixed training with multi-gpu. If you want to do mixed training, please use no more than 1 gpu by the command above.

### Evaluation
For evaluatation we follow the code mlp.py provided by ogb [here](https://github.com/snap-stanford/ogb/blob/master/examples/linkproppred/collab/mlp.py).

### Used config
ogbl-collab
```
python3 deepwalk.py --ogbl_name ogbl-collab --load_from_ogbl --save_in_pt --output_emb_file collab-embedding.pt --num_walks 50 --window_size 2 --walk_length 40 --lr 0.1 --negative 1 --neg_weight 1 --lap_norm 0.01 --mix --gpus 0 --num_threads 4 --print_interval 2000 --print_loss --batch_size 128 --use_context_weight
cd ./ogb/blob/master/examples/linkproppred/collab/
cp embedding_pt_file_path ./
python3 mlp.py --device 0 --runs 10 --use_node_embedding
```

ogbl-ddi
```
python3 deepwalk.py --ogbl_name ogbl-ddi --load_from_ogbl --save_in_pt --output_emb_file ddi-embedding.pt --num_walks 50 --window_size 2 --walk_length 80 --lr 0.1 --negative 1 --neg_weight 1 --lap_norm 0.05 --only_gpu --gpus 0 --num_threads 4 --print_interval 2000 --print_loss --batch_size 16 --use_context_weight
cd ./ogb/blob/master/examples/linkproppred/ddi/
cp embedding_pt_file_path ./
python3 mlp.py --device 0 --runs 10 --epochs 100
```

ogbl-ppa
```
python3 deepwalk.py --ogbl_name ogbl-ppa --load_from_ogbl --save_in_pt --output_emb_file ppa-embedding.pt --negative 1 --neg_weight 1 --batch_size 64 --print_interval 2000 --print_loss --window_size 1 --num_walks 30 --walk_length 80 --lr 0.1 --lap_norm 0.02 --mix --gpus 0 --num_threads 4
cp embedding_pt_file_path ./
python3 mlp.py --device 2 --runs 10
```

ogbl-citation
```
python3 deepwalk.py --ogbl_name ogbl-citation --load_from_ogbl --save_in_pt --output_emb_file embedding.pt --window_size 2 --num_walks 10 --negative 1 --neg_weight 1 --walk_length 80 --batch_size 128 --print_loss --print_interval 1000 --mix --gpus 0 --use_context_weight --num_threads 4 --lap_norm 0.01 --lr 0.1
cp embedding_pt_file_path ./
python3 mlp.py --device 2 --runs 10 --use_node_embedding
```

### OGBL Results
ogbl-collab
<br>#params: 61258346(model) + 131841(mlp) = 61390187
<br>Hits@10
<br>&emsp;Highest Train: 74.83 ± 4.79
<br>&emsp;Highest Valid: 40.03 ± 2.98
<br>&emsp;&emsp;Final Train: 74.51 ± 4.92
<br>&emsp;&emsp;Final Test: 31.13 ± 2.47
<br>Hits@50
<br>&emsp;Highest Train: 98.83 ± 0.15
<br>&emsp;Highest Valid: 60.61 ± 0.32
<br>&emsp;&emsp;Final Train: 98.74 ± 0.17
<br>&emsp;&emsp;Final Test: 50.37 ± 0.34
<br>Hits@100
<br>&emsp;Highest Train: 99.86 ± 0.04
<br>&emsp;Highest Valid: 66.64 ± 0.32
<br>&emsp;&emsp;Final Train: 99.84 ± 0.06
<br>&emsp;&emsp;Final Test: 56.88 ± 0.37

<br>obgl-ddi
<br>#params: 1444840(model) + 99073(mlp) = 1543913
<br>Hits@10
<br>&emsp;Highest Train: 33.91 ± 2.01
<br>&emsp;Highest Valid: 30.96 ± 1.89
<br>&emsp;&emsp;Final Train: 33.90 ± 2.00
<br>&emsp;&emsp;Final Test: 15.16 ± 4.28
<br>Hits@20
<br>&emsp;Highest Train: 44.64 ± 1.71
<br>&emsp;Highest Valid: 41.32 ± 1.69
<br>&emsp;&emsp;Final Train: 44.62 ± 1.69
<br>&emsp;&emsp;Final Test: 26.42 ± 6.10
<br>Hits@30
<br>&emsp;Highest Train: 51.01 ± 1.72
<br>&emsp;Highest Valid: 47.64 ± 1.71
<br>&emsp;&emsp;Final Train: 50.99 ± 1.72
<br>&emsp;&emsp;Final Test: 33.56 ± 3.95

<br>ogbl-ppa
<br>#params: 150024820(model) + 113921(mlp) = 150138741
<br>Hits@10
<br>&emsp;Highest Train: 4.78 ± 0.73
<br>&emsp;Highest Valid: 4.30 ± 0.68
<br>&emsp;&emsp;Final Train: 4.77 ± 0.73
<br>&emsp;&emsp;Final Test: 2.67 ± 0.42
<br>Hits@50
<br>&emsp;Highest Train: 18.82 ± 1.07
<br>&emsp;Highest Valid: 17.26 ± 1.01
<br>&emsp;&emsp;Final Train: 18.82 ± 1.07
<br>&emsp;&emsp;Final Test: 17.34 ± 2.09
<br>Hits@100
<br>&emsp;Highest Train: 31.29 ± 2.11
<br>&emsp;Highest Valid: 28.97 ± 1.92
<br>&emsp;&emsp;Final Train: 31.28 ± 2.12
<br>&emsp;&emsp;Final Test: 28.88 ± 1.53

<br>ogbl-citation
<br>#params: 757811178(model) + 131841(mlp) = 757943019
<br>MRR
<br>&emsp;Highest Train: 0.9381 ± 0.0003
<br>&emsp;Highest Valid: 0.8469 ± 0.0003
<br>&emsp;&emsp;Final Train: 0.9377 ± 0.0004
<br>&emsp;&emsp;Final Test: 0.8479 ± 0.0003

### Notes
#### Multi-GPU issues
For efficiency, the results of ogbl-collab, ogbl-ppa, ogbl-ddi are run with multi-GPU. Since ogb is somehow incompatible with our multi-GPU implementation, we need to do some preprocessing. The command is:
```
python3 load_dataset.py --name dataset_name
```
It will output a data file to the local. For example, if `dataset_name` is `ogbl-collab`, then a file `ogbl-collab-net.txt` will be generated. Then we run 
```
python3 deepwalk.py --data_file data_file_path
```
where the other parameters are the same with used configs without using `--load_from_ogbl` and `--ogbl_name`.

#### Others
The performance on ogbl-ddi and ogbl-ppa can be not that stable.