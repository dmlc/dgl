# Transformer in DGL
In this example we implement the [Transformer](https://arxiv.org/pdf/1706.03762.pdf) and [Universal Transformer](https://arxiv.org/abs/1807.03819) with ACT in DGL.

The folder contains training module and inferencing module (beam decoder) for Transformer and training module for Universal Transformer

## Dependencies

- PyTorch 0.4.1+
- networkx
- tqdm
- requests

## Usage

- For training:

    ```
    python3 translation_train.py [--gpus id1,id2,...] [--N #layers] [--dataset DATASET] [--batch BATCHSIZE] [--universal]
    ```

- For evaluating BLEU score on test set(by enabling `--print` to see translated text):

    ```
    python3 translation_test.py [--gpu id] [--N #layers] [--dataset DATASET] [--batch BATCHSIZE] [--checkpoint CHECKPOINT] [--print] [--universal]
    ```

Available datasets: `copy`, `sort`, `wmt14`, `multi30k`(default).

## Test Results

### Transformer

- Multi30k: we achieve BLEU score 35.41 with default setting on Multi30k dataset, without using pre-trained embeddings. (if we set the number of layers to 2, the BLEU score could reach 36.45).
- WMT14: work in progress 

### Universal Transformer

- work in progress 

## Notes

- Currently we do not support Multi-GPU training(this will be fixed soon), you should only specify only one gpu\_id when running the training script.

## Reference

- [The Annotated Transformer](http://nlp.seas.harvard.edu/2018/04/03/attention.html)
- [Tensor2Tensor](https://github.com/tensorflow/tensor2tensor/blob/master/tensor2tensor/)
