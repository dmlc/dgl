Representation Learning for Attributed Multiplex Heterogeneous Network (GANTE)
============

- Paper link: [https://arxiv.org/abs/1905.01669](https://arxiv.org/abs/1905.01669)
- Author's code repo: [https://github.com/THUDM/GATNE](https://github.com/THUDM/GATNE). Note that only GATNE-T is implemented here.

Requirements
------------
- requirements

``bash
pip install requirements
``

Datasets
--------
"example": [https://s3.us-west-2.amazonaws.com/dgl-data/dataset/recsys/GATNE/example.zip](https://s3.us-west-2.amazonaws.com/dgl-data/dataset/recsys/GATNE/example.zip)
"amazon": [https://s3.us-west-2.amazonaws.com/dgl-data/dataset/recsys/GATNE/amazon.zip](https://s3.us-west-2.amazonaws.com/dgl-data/dataset/recsys/GATNE/amazon.zip)
"youtube": [https://s3.us-west-2.amazonaws.com/dgl-data/dataset/recsys/GATNE/youtube.zip](https://s3.us-west-2.amazonaws.com/dgl-data/dataset/recsys/GATNE/youtube.zip)
"twitter": [https://s3.us-west-2.amazonaws.com/dgl-data/dataset/recsys/GATNE/twitter.zip](https://s3.us-west-2.amazonaws.com/dgl-data/dataset/recsys/GATNE/twitter.zip)


Training
--------

Run with following (available dataset: "example", "youtube", "amazon")
```bash
python src/main.py --input data/example
```

To run on "twitter" dataset, use
```bash
python src/main.py --input data/twitter --eval-type 1
```

