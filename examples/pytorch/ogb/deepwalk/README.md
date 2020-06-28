## How to load ogb data
To load ogb dataset, you need to run the following command, which will output a network file, ogbl-collab-net.txt:
```
python3 load_dataset.py --name ogbl-collab
```

## Evaluation
For evaluatation we follow the code mlp.py provided by ogb [here](https://github.com/snap-stanford/ogb/blob/master/examples/linkproppred/collab/mlp.py).

## Used config
ogbl-collab
```
python3 deepwalk.py --data_file ogbl-collab-net.txt --save_in_pt --output_emb_file embedding.pt --num_walks 50 --window_size 20 --walk_length 40 --lr 0.1 --negative 1 --neg_weight 1 --lap_norm 0.005 --mix --adam --gpus 0 1 2 3 --num_threads 4 --print_interval 2000 --print_loss --batch_size 32
cd ./ogb/blob/master/examples/linkproppred/collab/
cp embedding_pt_file_path ./
python3 mlp.py --device 0 --runs 10 --use_node_embedding
```

ogbl-ddi
```
python3 deepwalk.py --data_file ogbl-ddi-net.txt --save_in_pt --output_emb_file ddi-embedding.pt --num_walks 50 --window_size 2 --walk_length 80 --lr 0.1 --negative 1 --neg_weight 1 --lap_norm 0.05 --only_gpu --adam --gpus 0 --num_threads 4 --print_interval 2000 --print_loss --batch_size 16 --use_context_weight
cd ./ogb/blob/master/examples/linkproppred/ddi/
cp embedding_pt_file_path ./
python3 mlp.py --device 0 --runs 5
```

ogbl-ppa
```
python3 deepwalk.py --data_file ogbl-ppa-net.txt --save_in_pt --output_emb_file ppa-embedding.pt --negative 1 --neg_weight 1 --batch_size 64 --print_interval 2000 --print_loss --window_size 2 --num_walks 30 --walk_length 80 --lr 0.1 --lap_norm 0.02 --adam --mix --gpus 0 1 --use_context_weight --num_threads 4
cp embedding_pt_file_path ./
python3 mlp.py --device 2 --runs 10
```

ogbl-citation
```
python3 deepwalk.py --data_file ogbl-citation-net.txt --save_in_pt --output_emb_file embedding.pt --window_size 2 --num_walks 10 --negative 1 --neg_weight 1 --walk_length 80 --batch_size 128 --print_loss --print_interval 1000 --mix --adam --gpus 0 1 2 3 --use_context_weight --num_threads 4 --lap_norm 0.05 --lr 0.1
cp embedding_pt_file_path ./
python3 mlp.py --device 2 --runs 5 --use_node_embedding
```

## Score
ogbl-collab
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
<br>Hits@10
<br>&emsp;Highest Train: 35.05 ± 3.68
<br>&emsp;Highest Valid: 31.72 ± 3.52
<br>&emsp;&emsp;Final Train: 35.05 ± 3.68
<br>&emsp;&emsp;Final Test: 12.68 ± 3.19
<br>Hits@20
<br>&emsp;Highest Train: 44.85 ± 1.26
<br>&emsp;Highest Valid: 41.20 ± 1.41
<br>&emsp;&emsp;Final Train: 44.85 ± 1.26
<br>&emsp;&emsp;Final Test: 21.69 ± 3.14
<br>Hits@30
<br>&emsp;Highest Train: 52.28 ± 1.21
<br>&emsp;Highest Valid: 48.49 ± 1.09
<br>&emsp;&emsp;Final Train: 52.28 ± 1.21
<br>&emsp;&emsp;Final Test: 29.13 ± 3.46


<br>ogbl-ppa
<br>Hits@10
<br>&emsp;Highest Train: 3.58 ± 0.90
<br>&emsp;Highest Valid: 2.88 ± 0.76
<br>&emsp;&emsp;Final Train: 3.58 ± 0.90
<br>&emsp;&emsp;Final Test: 1.45 ± 0.65
<br>&emsp;Hits@50
<br>&emsp;Highest Train: 18.21 ± 2.29
<br>&emsp;Highest Valid: 15.75 ± 2.10
<br>&emsp;&emsp;Final Train: 18.21 ± 2.29
<br>&emsp;&emsp;Final Test: 11.70 ± 0.97
<br>&emsp;Hits@100
<br>&emsp;Highest Train: 31.16 ± 2.23
<br>&emsp;Highest Valid: 27.52 ± 2.07
<br>&emsp;&emsp;Final Train: 31.16 ± 2.23
<br>&emsp;&emsp;Final Test: 23.02 ± 1.63

<br>ogbl-citation
<br>MRR
<br>&emsp;Highest Train: 0.8796 ± 0.0007
<br>&emsp;Highest Valid: 0.8141 ± 0.0007
<br>&emsp;&emsp;Final Train: 0.8793 ± 0.0008
<br>&emsp;&emsp;Final Test: 0.8159 ± 0.0006
