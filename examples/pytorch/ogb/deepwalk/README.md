# DeepWalk

- Paper link: [here](https://arxiv.org/pdf/1403.6652.pdf)
- Other implementation: [gensim](https://github.com/phanein/deepwalk), [deepwalk-c](https://github.com/xgfs/deepwalk-c)

The implementation includes multi-processing training with CPU and mixed training with CPU and multi-GPU.

## Dependencies
- PyTorch 1.0.1+

## Tested version
- PyTorch 1.5.0
- DGL 0.4.3
- OGB 1.2.0

## How to load ogb data
To load ogb dataset, you need to run the following command, which will output a network file, ogbl-collab-net.txt:
```
python3 load_dataset.py --name ogbl-collab
```

For other datasets please pass the full path to the trainer through --data\_file and the format of a network file should follow:
```
1(node id) 2(node id)
1 3
1 4
2 4
...
```

## How to run the code
To run the code:
```
python3 deepwalk.py --data_file youtube --output_emb_file emb.txt --adam --mix --lr 0.2 --gpus 0 1 2 3 --batch_size 100 --negative 5
```

## How to save the embedding
By default the trained embedding is saved under --output\_embe\_file FILE\_NAME as a numpy object.
To save the trained embedding in raw format(txt format), please use --save\_in\_txt argument.
We also provide To save the trained embedding in raw format(txt format), please use --save\_in\_txt argument.

## Evaluation
For evaluatation we follow the code provided by ogb [here](https://github.com/snap-stanford/ogb/blob/master/examples/linkproppred/collab/mlp.py).

Recommended config:
```
python3 deepwalk.py --data_file ogbl-collab-net.txt --save_in_pt --output_emb_file embedding.pt --num_walks 70 --window_size 10 --walk_length 40 --lr 0.1 --negative 1 --neg_weight 1 --lap_norm 0.01 --mix --adam --gpus 0 1 2 3 --num_threads 4 --print_interval 2000 --batch_size 32
```