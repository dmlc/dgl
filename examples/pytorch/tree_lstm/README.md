# Tree-LSTM
This is a re-implementation of the following paper:

> [**Improved Semantic Representations From Tree-Structured Long Short-Term Memory Networks**](http://arxiv.org/abs/1503.00075) 
> *Kai Sheng Tai, Richard Socher, and Christopher Manning*. 

The provided implementation can achieve a test accuracy of 50.59 which is comparable with the result reported in the paper 51.0.

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

## Speed Test
To enable fair comparison with [DyNet Tree-LSTM implementation](https://github.com/clab/dynet/tree/master/examples/treelstm), we set the batch size to 100.
```
python train.py --gpu 0 --batch-size 100
```

| Device              | Framework | Speed(time per batch) |
|---------------------|-----------|-----------------------|
| GeForce GTX TITAN X | DGL       | 7.23(±0.66)s          |
