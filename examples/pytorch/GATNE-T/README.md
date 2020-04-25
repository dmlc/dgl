Representation Learning for Attributed Multiplex Heterogeneous Network (GANTE)
============

- Paper link: [https://arxiv.org/abs/1905.01669](https://arxiv.org/abs/1905.01669)
- Author's code repo: [https://github.com/THUDM/GATNE](https://github.com/THUDM/GATNE). Note that only GATNE-T is implemented here.

Requirements
------------
- requests

``bash
pip install requests
``


Training
-------

Run with following (available dataset: "example", "youtube", "amazon")
```bash
python src/main.py --input data/example
```

To run on "twitter" dataset, use
```bash
python src/main.py --input data/twitter --eval-type 1
```

