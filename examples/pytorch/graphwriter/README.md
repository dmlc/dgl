# GraphWriter-DGL
In this example we implement the GraphWriter, [Text Generation from Knowledge Graphs with Graph Transformers](https://arxiv.org/abs/1904.02342) in DGL. And the [author's code](https://github.com/rikdz/GraphWriter). 

## Dependencies
- PyTorch >= 1.2  
- tqdm   
- pycoco (only for testing)  
- multi-bleu.perl and other scripts from mosesdecoder (only for testing)

## Usage
```
  # download data
  sh prepare_data.sh 
  # training
  sh run.sh
  # testing
  sh test.sh
```

## Result on AGENDA
| |BLEU|METEOR| training time per epoch|
|-|-|-|-|
|Author's implementation|14.3+-1.01| 18.8+-0.28| 1970s|
|DGL implementation|14.31+-0.34|19.74+-0.69| 1080s|

We use the author's code for the speed test, and our testbed is V100 GPU.

| |BLEU| detok BLEU| METEOR | 
|-|-|-|-|
|greedy, two layers| 13.97 +- 0.40| 13.78 +- 0.46| 18.76 +- 0.36|
|beam 4, length penalty 1.0, two layers| 14.66 +- 0.65| 14.53 +- 0.52| 19.50 +- 0.49|
|beam 4, length penalty 0.0, two layers| 14.33 +- 0.39| 14.09 +- 0.39| 18.63 +- 0.52|
|greedy, six layers| 14.17 +- 0.46| 14.01 +- 0.51| 19.18 +- 0.49|
|beam 4, length penalty 1.0, six layers| 14.31 +- 0.34| 14.35 +- 0.36| 19.74 +- 0.69|
|beam 4, length penalty 0.0, six layers| 14.40 +- 0.85| 14.15 +- 0.84| 18.86 +- 0.78|

We repeat the experiment five times. 

### Examples

We also provide the output of our implementation on test set together with the reference text.
- [GraphWriter's output](https://data.dgl.ai/models/graphwriter/tmp_pred.txt)
- [Reference text](https://data.dgl.ai/models/graphwriter/tmp_gold.txt)

