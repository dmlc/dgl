# (Universal) Transformer DGL/PyTorch

Reproduction of (Universal) Transformer in DGL.

## Usage

    python translation_train.py [--gpus id1,id2,...] [--N #layers] [--dataset DATASET] [--batch BATCHSIZE]
    python translation_test.py [--gpu id] [--N #layers] [--dataset DATASET] [--batch BATCHSIZE] [--checkpoint CHECKPOINT] [--print]

## Results

Multi30k: we achieve BLEU score 35.10 on Multi30k dataset, without using pre-trained embeddings.

## TODOs

- [ ] MultiGPU support (lack of scheduling lock)

## Reference

- [The Annotated Transformer](http://nlp.seas.harvard.edu/2018/04/03/attention.html)
- [universal\_transformer.py](https://github.com/tensorflow/tensor2tensor/blob/master/tensor2tensor/models/research/universal_transformer.py)
