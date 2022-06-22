Graph Isomorphism Network (GIN)
============

- Paper link: [arXiv](https://arxiv.org/abs/1810.00826) [OpenReview](https://openreview.net/forum?id=ryGs6iA5Km) 
- Author's code repo: [https://github.com/weihua916/powerful-gnns](https://github.com/weihua916/powerful-gnns).

Dependencies
------------
- PyTorch 1.1.0+
- sklearn
- tqdm

``bash
pip install torch sklearn tqdm
``

How to run
----------

An experiment on the GIN in default settings can be run with

```bash
python main.py
```

An experiment on the GIN in customized settings can be run with
```bash
python main.py [--device 0 | --disable-cuda] --dataset COLLAB \
               --graph_pooling_type max --neighbor_pooling_type sum
```
add `--degree_as_nlabel` to use one-hot encodings of node degrees as node feature vectors

Results
-------

results may **fluctuate**, due to random factors and the relatively small data set. if you want to follow the paper's setting, consider the script below.

```bash
# 4 bioinformatics datasets setting graph_pooling_type=sum, the nodes have categorical input features 
python main.py --dataset MUTAG --device 0  \
                --graph_pooling_type sum --neighbor_pooling_type sum --filename MUTAG.txt

python main.py --dataset PTC --device 0  \
                --graph_pooling_type sum --neighbor_pooling_type sum --filename PTC.txt

python main.py --dataset NCI1 --device 0  \
                --graph_pooling_type sum --neighbor_pooling_type sum --filename NCI1.txt

python main.py --dataset PROTEINS --device 0  \
                --graph_pooling_type sum --neighbor_pooling_type sum --filename PROTEINS.txt

# 5 social network datasets setting graph_pooling_type=mean, for the REDDIT datasets, we set all node feature vectors to be the same 
# (thus, features here are uninformative); for the other social networks, we use one-hot encodings of node degrees.  
python main.py --dataset COLLAB --device 0  \
                --graph_pooling_type mean --neighbor_pooling_type sum --degree_as_nlabel --filename COLLAB.txt

python main.py --dataset IMDBBINARY --device 0  \
                --graph_pooling_type mean --neighbor_pooling_type sum --degree_as_nlabel --filename IMDBBINARY.txt

python main.py --dataset IMDBMULTI --device 0  \
                --graph_pooling_type mean --neighbor_pooling_type sum --degree_as_nlabel --filename IMDBMULTI.txt

python main.py --dataset REDDITBINARY --device 0  \
                --graph_pooling_type mean --neighbor_pooling_type sum --filename REDDITBINARY.txt --fold_idx 6 --epoch 120

python main.py --dataset REDDITMULTI5K --device 0  \
                --graph_pooling_type mean --neighbor_pooling_type sum --filename REDDITMULTI5K.txt
```

one fold of 10 result are below.

| dataset       | our result | paper report |
| ------------- | ---------- | ------------ |
| MUTAG         | 89.4      | 89.4 ± 5.6   |
| PTC           | 68.5      | 64.6 ± 7.0   |
| NCI1          | 78.5      | 82.7 ± 1.7   |
| PROTEINS      | 72.3      | 76.2 ± 2.8   |
| COLLAB        | 81.6      | 80.2 ± 1.9   |
| IMDBBINARY    | 73.0      | 75.1 ± 5.1   |
| IMDBMULTI     | 54.0      | 52.3 ± 2.8   |
| REDDITBINARY  | 88.0      | 92.4 ± 2.5   |
| REDDITMULTI5K | 54.8      | 57.5 ± 1.5   |

