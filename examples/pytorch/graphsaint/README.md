# GraphSAINT

This DGL example implements the paper: GraphSAINT: Graph Sampling Based Inductive Learning Method.

Paper link: https://arxiv.org/abs/1907.04931

Author's code: https://github.com/GraphSAINT/GraphSAINT

Contributor: Jiahang Li ([@ljh1064126026](https://github.com/ljh1064126026))  Tang Liu ([@lt610](https://github.com/lt610))

## Dependencies

- PyTorch 1.8.1
- NumPy 1.19.2
- Scikit-learn 0.23.2
- DGL 0.8.1

## Dataset

All datasets used are provided by Author's [code](https://github.com/GraphSAINT/GraphSAINT). They are available in [Google Drive](https://drive.google.com/drive/folders/1zycmmDES39zVlbVCYs88JTJ1Wm5FbfLz) (alternatively, [Baidu Wangpan (code: f1ao)](https://pan.baidu.com/s/1SOb0SiSAXavwAcNqkttwcg#list/path=%2F)).

For google drive, you can proceed as follows

```bash
pip install gdown
gdown --folder URL
```

where URL is for a single dataset.

Dataset summary("m" stands for multi-label binary classification, and "s" for single-label.):
| Dataset | Nodes | Edges | Degree | Feature | Classes |
| :-: | :-: | :-: | :-: | :-: | :-: |
| PPI | 14,755 | 225,270 | 15 | 50 | 121(m) |
| Flickr | 89,250 | 899,756 | 10 | 500 | 7(s) |
| Reddit | 232,965 | 11,606,919 | 50 | 602 | 41(s) |
| Yelp | 716,847 | 6,977,410 | 10 | 300 | 100 (m) |
| Amazon | 1,598,960 | 132,169,734 | 83 | 200 | 107 (m) |

Note that the PPI dataset here is different from DGL's built-in variant.

## Minibatch training

Run with following:
```bash
python train_sampling.py --task $task
```

- `$task` includes `ppi_n, ppi_e, ppi_rw, flickr_n, flickr_e, flickr_rw, reddit_n, reddit_e, reddit_rw, yelp_n, yelp_e, yelp_rw, amazon_n, amazon_e, amazon_rw`. For example, `ppi_n` represents running experiments on dataset `ppi` with `node sampler`
