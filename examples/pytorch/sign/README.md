SIGN: Scalable Inception Graph Neural Networks
===============

- paper link: [https://arxiv.org/pdf/2004.11198.pdf](https://arxiv.org/pdf/2004.11198.pdf)

Requirements
----------------

```bash
pip install requests ogb
```

Results
---------------
### [Ogbn-products](https://ogb.stanford.edu/docs/nodeprop/#ogbn-products) (Amazon co-purchase dataset)

```bash
python sign.py --dataset amazon
```

Test accuracy: mean 0.78672, std 0.00059

### Reddit
```bash
python sign.py --dataset reddit
```

Test accuracy: mean 0.96326, std 0.00010
