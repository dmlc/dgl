# DAGNN

### Performance

#### On Cora, Citeseer and Pubmed
##### Accuracy
| Dataset | Cora | Citeseer | Pubmed |
| :-: | :-: | :-: | :-: |
| DAGNN | 84.4 ± 0.5 | 73.3 ± 0.6 | 80.5 ± 0.5 |
| DGL | 84.3 ± 0.6 | 73.5 ± 0.8 | 80.5 ± 0.3 |

#### On Arxiv
##### Accuracy
| Dataset | Arxiv |
| :-: | :-: |
| DAGNN | 72.09 ± 0.25 |
| DGL | 72.39 ± 0.23 |


#### On Gowalla, Yelp2018 and Amanzon-Book

Gowalla, Yelp2018 and Amanzon-Book are datasets in the filed of recommendation systems, which are introduced in paper [Neural Graph Collaborative Filtering](https://arxiv.org/abs/1905.08108v1)

##### Recall
| Dataset | Gowalla | Yelp2018 | Amanzon-Book |
| :-: | :-: | :-: | :-: |
| NGCF | 0.1556 | 0.0543 | 0.0313 |
| DAGNN(DGL) | 0.1792 | 0.0598 | 0.0408 |

##### NDCG
| Dataset | Gowalla | Yelp2018 | Amanzon-Book |
| :-: | :-: | :-: | :-: |
| NGCF | 0.1327 | 0.0477 | 0.0263 |
| DAGNN(DGL) | 0.1507 | 0.0493 | 0.0315 |
