# Tree-LSTM
This is a re-implementation of the following paper:

> [**Improved Semantic Representations From Tree-Structured Long Short-Term Memory Networks**](http://arxiv.org/abs/1503.00075)
> *Kai Sheng Tai, Richard Socher, and Christopher Manning*.

The provided implementation can achieve a test accuracy of 51.72 which is comparable with the result reported in the original paper: 51.0(±0.5).

## Dependencies
* MXNet nightly build
* requests
* nltk

```bash
pip install mxnet --pre
pip install requests nltk
```

## Data
The script will download the [SST dataset] (http://nlp.stanford.edu/sentiment/index.html) and the GloVe 840B.300d embedding automatically if `--use-glove` is specified (note: download may take a while).

## Usage
```
DGLBACKEND=mxnet python3 train.py --gpu 0
```

## Speed Test

See https://docs.google.com/spreadsheets/d/1eCQrVn7g0uWriz63EbEDdes2ksMdKdlbWMyT8PSU4rc .

## Note
The code can work with MXNet 1.5.1
