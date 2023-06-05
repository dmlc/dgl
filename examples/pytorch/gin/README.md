Graph Isomorphism Network (GIN)
============

- Paper link: [arXiv](https://arxiv.org/abs/1810.00826) [OpenReview](https://openreview.net/forum?id=ryGs6iA5Km) 
- Author's code repo: [https://github.com/weihua916/powerful-gnns](https://github.com/weihua916/powerful-gnns).

Dependencies
------------
- scikit-learn

Install as follows:
```bash
pip install scikit-learn
```

How to run
-------

Run with the following for bioinformatics graph classification (available datasets: MUTAG (default), PTC, NCI1, and PROTEINS)
```bash
python3 train.py --dataset MUTAG
```

> **_NOTE:_**  Users may observe results fluctuate due to the randomness with relatively small dataset.  In consistence with the original [paper](https://arxiv.org/abs/1810.00826), five social network datasets, 'COLLAB', 'IMDBBINARY' 'IMDBMULTI' 'REDDITBINARY' and 'REDDITMULTI5K', are also available as the input. Users are encouraged to update the script slightly for social network applications, for example, replacing sum readout on bioinformatics datasets with mean readout on social network datasets and using one-hot encodings of node degrees by setting "degree_as_nlabel=True" in GINDataset.

Summary (10-fold cross-validation)
-------
| Dataset       | Result
| ------------- | -------
| MUTAG         | ~89.4
| PTC           | ~68.5
| NCI1          | ~82.9
| PROTEINS      | ~74.1
