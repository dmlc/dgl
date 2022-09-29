Metapath2vec
============

- Paper link: [metapath2vec: Scalable Representation Learning for Heterogeneous Networks](https://ericdongyx.github.io/papers/KDD17-dong-chawla-swami-metapath2vec.pdf)
- Author's code repo: [https://ericdongyx.github.io/metapath2vec/m2v.html](https://ericdongyx.github.io/metapath2vec/m2v.html). 

Dependencies
------------
- PyTorch 1.0.1+

How to run the code
-----
Run with either of the following procedures:

* Running with default AMiner dataset:
  1. Directly run the following command:

     ```bash
     python metapath2vec.py --aminer --path "where/you/want/to/download" --output_file "your_model_output_path"
     ```
* Running with another AMiner-like dataset
  1. Prepare the data in the same format as the ones of AMiner and DBIS in Section B of [Author's code repo](https://ericdongyx.github.io/metapath2vec/m2v.html).
  2. Run `sampler.py` on your graph dataset with, for instance,

     ```bash
     python sampler.py net_dbis
     ```
  3. Run the following command:

     ```bash
     python metapath2vec.py --path net_dbis/output_path.txt --output_file "your_model_output_path"
     ```

Tips: Change num_workers based on your GPU instances; Running 3 or 4 epochs is actually enough. 

Tricks included in the implementation:
-------
1, Sub-sampling;

2, Negative Sampling without repeatedly calling numpy random choices;

Performance and Explanations:
-------
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

Testing files are available in "label 2" file;

The above are results listed in the paper, in real experiments, exact numbers might be slightly different:

1, For venue node classification results, when the size of the training dataset is small (e.g. 5%), the variance of the performance is large since the number of available labeled venues is small. 

2, For author node classification results, the performance is stable since the number of available labeled authors is huge, so even 5% training data would be sufficient.

3, In the test.py, you could change experiment times you want, especially it is very slow to test author classification so you could only do 1 or 2 times.
