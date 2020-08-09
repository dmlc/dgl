# Transformer in DGL

**This example is out-dated, please refer to [BP-Transformer](http://github.com/yzh119/bpt) for efficient (Sparse) Transformer implementation in DGL.**

In this example we implement the [Transformer](https://arxiv.org/pdf/1706.03762.pdf) with ACT in DGL.

The folder contains training module and inferencing module (beam decoder) for Transformer.

## Dependencies

- PyTorch 0.4.1+
- networkx
- tqdm
- requests
- matplotlib

## Usage

- For training:

    ```
    python3 translation_train.py [--gpus id1,id2,...] [--N #layers] [--dataset DATASET] [--batch BATCHSIZE] [--universal]
    ```

By specifying multiple gpu ids separated by comma, we will employ multi-gpu training with multiprocessing.

- For evaluating BLEU score on test set(by enabling `--print` to see translated text):

    ```
    python3 translation_test.py [--gpu id] [--N #layers] [--dataset DATASET] [--batch BATCHSIZE] [--checkpoint CHECKPOINT] [--print] [--universal]
    ```

Available datasets: `copy`, `sort`, `wmt14`, `multi30k`(default).

## Test Results

- Multi30k: we achieve BLEU score 35.41 with default setting on Multi30k dataset, without using pre-trained embeddings. (if we set the number of layers to 2, the BLEU score could reach 36.45).
- WMT14: work in progress 

## Reference

- [The Annotated Transformer](http://nlp.seas.harvard.edu/2018/04/03/attention.html)
- [Tensor2Tensor](https://github.com/tensorflow/tensor2tensor/blob/master/tensor2tensor/)
