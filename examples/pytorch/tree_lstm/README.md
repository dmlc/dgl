# Tree-LSTM
This is a re-implementation of the following paper:

> [**Improved Semantic Representations From Tree-Structured Long Short-Term Memory Networks**](http://arxiv.org/abs/1503.00075) 
> *Kai Sheng Tai, Richard Socher, and Christopher Manning*. 

The provided implementation can achieve a test accuracy of 51.72 which is comparable with the result reported in the original paper: 51.0(±0.5).

## Data
The script will download the [SST dataset] (http://nlp.stanford.edu/sentiment/index.html) automatically, and you need to download the GloVe word vectors yourself. For the command line, you can use this.
```
wget http://nlp.stanford.edu/data/glove.840B.300d.zip
unzip glove.840B.300d.zip
```

## Dependencies
* PyTorch 0.4.1+
* requests
* nltk

```
pip install torch requests nltk
```

## Usage
```
python3 train.py --gpu 0
```

## Speed

On AWS p3.2x instance, it can achieve 3.18s per epoch when setting batch size to 256.
