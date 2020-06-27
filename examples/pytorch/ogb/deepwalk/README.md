## How to load ogb data
To load ogb dataset, you need to run the following command, which will output a network file, ogbl-collab-net.txt:
```
python3 load_dataset.py --name ogbl-collab
```

## Evaluation
For evaluatation we follow the code provided by ogb [here](https://github.com/snap-stanford/ogb/blob/master/examples/linkproppred/collab/mlp.py).
```
cd ./ogb/blob/master/examples/linkproppred/collab/
cp embedding_pt_file_path ./
python3 mlp.py --device 0 --runs 10 --use_node_embedding
```

## Used config
```
python3 deepwalk.py --data_file ogbl-collab-net.txt --save_in_pt --output_emb_file embedding.pt --num_walks 50 --window_size 20 --walk_length 40 --lr 0.1 --negative 1 --neg_weight 1 --lap_norm 0.005 --mix --adam --gpus 0 1 2 3 --num_threads 4 --print_interval 2000 --print_loss --batch_size 32
```


## Score
Hits@10
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