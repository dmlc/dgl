Metapath2vec
============

- Paper link: [metapath2vec: Scalable Representation Learning for Heterogeneous Networks](https://ericdongyx.github.io/papers/KDD17-dong-chawla-swami-metapath2vec.pdf)
- Author's code repo: [https://ericdongyx.github.io/metapath2vec/m2v.html](https://ericdongyx.github.io/metapath2vec/m2v.html). 

Dependencies
------------
- PyTorch 1.0.1+

How to run the code
-----
Run with the following procedures:

1, Run sampler.py on your graph dataset. Note that: the input text file should be list of mappings so you probably need to preprocess your graph dataset. Files with sample format are available in "net_dbis" file. Of course you could also use your own metapath sampler implementation.

2, Run the following command:
```bash
python metapath2vec.py --input_file "aminer.txt" --output_file "your_output_file_path"
```
aminer.txt file is available here: [https://drive.google.com/drive/my-drive]

Tips: Adapt num_workers based on your GPU instances; Running 3 or 4 epochs is actually enough. 

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
