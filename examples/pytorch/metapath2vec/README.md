# Metapath2vec

- Paper link: [metapath2vec: Scalable Representation Learning for Heterogeneous Networks](https://ericdongyx.github.io/papers/KDD17-dong-chawla-swami-metapath2vec.pdf)
- Authors' code: [https://ericdongyx.github.io/metapath2vec/m2v.html](https://ericdongyx.github.io/metapath2vec/m2v.html).

## Usage

### AMiner dataset

Download and unzip the AMiner dataset

```bash
wget https://www.dropbox.com/s/1bnz8r7mofx0osf/net_aminer.zip
unzip net_aminer.zip
```

Then run with

```bash
python metapath2vec.py --input_path net_aminer --output_file model
```

### AMiner-like custom dataset

Prepare the data in the format of DGLGraph. You can refer to  `reading_data.py` and construct your own graph and id-label maps if needed.

```bash
python metapath2vec.py --input_path "where/you/store/the/data" --meta_path [etypeA,etypeB] --output_file model
```

## Tricks included in the implementation:

1, Sub-sampling;

2, Negative Sampling without repeatedly calling numpy random choices;

## Performance and Explanations:

Venue Classification Results for Metapath2vec:

| Metric | 5% | 10% | 20% | 30% | 40% | 50% | 60% | 70% | 80% | 90% |
| ------ | -- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Macro-F1 | 0.3033 | 0.5247 | 0.8033 | 0.8971 | 0.9406 | 0.9532 | 0.9529 | 0.9701 | 0.9683 | 0.9670 |
| Micro-F1 | 0.4173 | 0.5975 | 0.8327 | 0.9011 | 0.9400 | 0.9522 | 0.9537 | 0.9725 | 0.9815 | 0.9857 |

Author Classfication Results for Metapath2vec:

| Metric | 5% | 10% | 20% | 30% | 40% | 50% | 60% | 70% | 80% | 90% |
| ------ | -- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Macro-F1 | 0.9216 | 0.9262 | 0.9292 | 0.9303 | 0.9309 | 0.9314 | 0.9315 | 0.9316 | 0.9319 | 0.9320 |
| Micro-F1 | 0.9279 | 0.9319 | 0.9346 | 0.9356 | 0.9361 | 0.9365 | 0.9365 | 0.9365 | 0.9367 | 0.9369 |

Note that:

Testing files are available in author's code repo: [https://ericdongyx.github.io/metapath2vec/m2v.html](https://ericdongyx.github.io/metapath2vec/m2v.html) [F] Ground Truth Labeled by Google Scholar Metrics 2016 for Multi-Label Node Classification and Clustering. You need to use `nid2word` parameter to map the node labels with the global embedding index.

The above are results listed in the paper, in real experiments, exact numbers might be slightly different:

1. For venue node classification results, when the size of the training dataset is small (e.g. 5%), the variance of the performance is large since the number of available labeled venues is small.

2. For author node classification results, the performance is stable since the number of available labeled authors is huge, so even 5% training data would be sufficient.

3. In the test.py, you could change experiment times you want, especially it is very slow to test author classification so you could only do 1 or 2 times.

Set the `min_count` parameter to the appropriate size. Or there would be lots of discarded embeddings as last, which causes waste of resources.