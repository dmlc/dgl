# Tree-LSTM
This is a re-implementation of the following paper:

> [**Improved Semantic Representations From Tree-Structured Long Short-Term Memory Networks**](http://arxiv.org/abs/1503.00075) 
> *Kai Sheng Tai, Richard Socher, and Christopher Manning*. 

The provided implementation can achieve a test accuracy of 50.09 which is comparable with the result reported in the paper 51.0.
## Data
The script will download the [SST dataset] (http://nlp.stanford.edu/sentiment/index.html) automatically, and you need to download the GloVe word vectors yourself. For the command line, you can use this.
```
wget http://nlp.stanford.edu/data/glove.840B.300d.zip
unzip glove.840B.300d.zip
```

## Usage
```
python train.py --gpu 0
```