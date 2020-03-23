# Evaluate

It provide a python API to evaluate node embedding on multi-label classification with [LIBLINEAR](https://github.com/cjlin1/liblinear).

## How to run

Run with the following procedures:

1) compile the code:
```bash
sh make.sh
```

2) normalize node embedding:
```bash
./norm -input emb.txt -output emb.bin -binary 1
```

Format of emb.txt:
```
10312(number of nodes) 128(embedding dimension)
1(node id) 0.3000 0.4000 ...
2 0.5000 0.6000 ...
...
```

3) evaluate:
```bash
python2 test.py --emb emb.bin --vocab vocab.txt --label label.txt --portion 0.1
```
