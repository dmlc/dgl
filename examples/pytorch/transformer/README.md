# Transformer in DGL
In this example we implement the [Transformer](https://arxiv.org/pdf/1706.03762.pdf) and [Universal Transformer](https://arxiv.org/abs/1807.03819) in DGL.

## Requirements

- PyTorch 0.4.1+
- tqdm

## Usage

For training:

    python translation_train.py [--gpus id1,id2,...] [--N #layers] [--dataset DATASET] [--batch BATCHSIZE] [--universal]

For testing:

    python translation_test.py [--gpu id] [--N #layers] [--dataset DATASET] [--batch BATCHSIZE] [--checkpoint CHECKPOINT] [--print]

## Results

Multi30k: we achieve BLEU score 35.15 on Multi30k dataset, without using pre-trained embeddings.

## Notes

- Currently we do not support Multi-GPU training(this will be fixed soon), you should only specifiy only one gpu\_id when running the training script.

## Reference

- [The Annotated Transformer](http://nlp.seas.harvard.edu/2018/04/03/attention.html)
- [Tensor2Tensor](https://github.com/tensorflow/tensor2tensor/blob/master/tensor2tensor/)
