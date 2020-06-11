# DeepWalk

- Paper link: [here](https://arxiv.org/pdf/1403.6652.pdf)
- Other implementation: [gensim](https://github.com/phanein/deepwalk), [deepwalk-c](https://github.com/xgfs/deepwalk-c)

The implementation includes multi-processing training with CPU and mixed training with CPU and multi-GPU.

## Dependencies
- PyTorch 1.0.1+

## Tested version
- PyTorch 1.5.0
- DGL 0.4.3


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

## How to run the code
To run the code:
```
python3 deepwalk.py --data_file youtube --output_emb_file emb.txt --adam --mix --lr 0.2 --gpus 0 1 2 3 --batch_size 100 --negative 5
```

## How to save the embedding
By default the trained embedding is saved under --output\_embe\_file FILE\_NAME as a numpy object.
To save the trained embedding in raw format(txt format), please use --save\_in\_txt argument.

## Evaluation

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
