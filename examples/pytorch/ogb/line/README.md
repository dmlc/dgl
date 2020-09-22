# LINE Example
- Paper link: [here](https://arxiv.org/pdf/1503.03578)
- Official implementation: [here](https://github.com/tangjianpku/LINE)

This implementation includes both LINE-1st and LINE-2nd. The detailed usage is shown in the arguments in line.py.

## How to load ogb data
To load ogb dataset, you need to run the following command, which will output a network file, ogbn-products-net.txt:
```
python3 load_dataset.py --name ogbn-proteins
```
Or you can run the code directly with:
```
python3 line.py --ogbn_name xxx --load_from_ogbn
```
However, ogb.nodeproppred might not be compatible with mixed training with multi-gpu. If you want to do mixed training, please use no more than 1 gpu by the command above. We leave the commands to run with multi-gpu at the end.

## Evaluation
For evaluatation we follow the code mlp.py provided by ogb [here](https://github.com/snap-stanford/ogb/blob/master/examples/nodeproppred/).

## Used config
ogbn-arxiv
```
python3 line.py --save_in_pt --dim 128 --lap_norm 0.1 --mix --gpus 0 --batch_size 1024 --output_emb_file arxiv-embedding.pt --num_samples 1000 --print_interval 1000 --negative 5 --fast_neg --load_from_ogbn --ogbn_name ogbn-arxiv
cd ./ogb/blob/master/examples/nodeproppred/arxiv
cp embedding_pt_file_path ./
python3 mlp.py --device 0 --use_node_embedding
```

ogbn-proteins
```
python3 line.py --save_in_pt --dim 128 --lap_norm 0.01 --mix --gpus 1 --batch_size 1024 --output_emb_file protein-embedding.pt --num_samples 600 --print_interval 1000 --negative 1 --fast_neg --load_from_ogbn --ogbn_name ogbn-proteins --print_loss
cd ./ogb/blob/master/examples/nodeproppred/proteins
cp embedding_pt_file_path ./
python3 mlp.py --device 0 --use_node_embedding
```

ogbl-products
```
python3 line.py --save_in_pt --dim 128 --lap_norm 0.01 --mix --gpus 0 --batch_size 4096 --output_emb_file products-embedding.pt --num_samples 3000 --print_interval 1000 --negative 1 --fast_neg --load_from_ogbn --ogbn_name ogbn-products --print_loss
cd ./ogb/blob/master/examples/nodeproppred/products
cp embedding_pt_file_path ./
python3 mlp.py --device 0 --use_node_embedding
```

## Results
ogbn-arxiv
<br>#params: 33023343(model) + 142888(mlp) = 33166231
<br>Highest Train: 82.94 ± 0.11
<br>Highest Valid: 71.76 ± 0.08
<br>Final Train: 80.74 ± 1.30
<br>Final Test: 70.47 ± 0.19

<br>obgn-proteins
<br>#params: 25853524(model) + 129648(mlp) = 25983172
<br>Highest Train: 93.11 ± 0.04
<br>Highest Valid: 70.50 ± 1.29
<br>Final Train: 77.66 ± 10.27
<br>Final Test: 62.07 ± 1.25

<br>ogbn-products
<br>#params: 477570049(model) + 136495(mlp) = 477706544
<br>Highest Train: 98.01 ± 0.32
<br>Highest Valid: 89.57 ± 0.09
<br>Final Train: 94.96 ± 0.43
<br>Final Test: 72.52 ± 0.29

## Notes
To utlize multi-GPU training, we need to load datasets as a local file before training by the following command:
```
python3 load_dataset.py --name dataset_name
```
where `dataset_name` can be `ogbn-arxiv`, `ogbn-proteins`, and `ogbn-products`. After that, a local file `$dataset_name$-graph.bin` will be generated. Then run:
```
python3 line.py --data_file $dataset_name$-graph.bin
```
where the other parameters are the same with used configs without using `--load_from_ogbn` and `--ogbn_name`.