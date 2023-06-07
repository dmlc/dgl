Spatio-Temporal Graph Convolutional Networks
============

- Paper link: [arXiv](https://arxiv.org/pdf/1709.04875v4.pdf)
- Author's code repo: https://github.com/VeritasYin/STGCN_IJCAI-18.
- See [this blog](https://towardsdatascience.com/build-your-first-graph-neural-network-model-to-predict-traffic-speed-in-20-minutes-b593f8f838e5) for more details about running the code.
- Dependencies
  - PyTorch 1.1.0+
  - scikit-learn
  - dgl
  - tables


How to run
----------
please get METR_LA dataset from [this Google drive](https://drive.google.com/open?id=10FOTa6HXPqX8Pf5WRoRwcFnW9BrNZEIX).
and [this Github repo](https://github.com/chnsh/DCRNN_PyTorch)

An experiment in default settings can be run with

```bash
python main.py
```

An experiment on the METR_LA dataset in customized settings can be run with
```bash
python main.py --lr --seed --disable-cuda --batch_size <batch-size> --epochs <number-of-epochs>
```

If one wishes to adjust the model structure, you can change the arguments `control_str` and `channels`
```bash
python main.py --control_str <control-string> --channels <n-input-channel> <n-hidden-channels-1> <n-hidden-channels-2> ... <n-output-channels>
```

`<control-string>` is a string of the following characters representing a sequence of neural network modules:

* `T`: representing a dilated temporal convolution layer, working on the temporal dimension.  The dilation factor is always twice as much as the previous temporal convolution layer.
* `S`: representing a graph convolution layer, working on the spatial dimension.  The input channels and output channels are the same.
* `N`: a Layer Normalization.

The argument list following `--channels` represents the output channels on each temporal convolution layer.  The list should have `N + 1` elements, where `N` is the number of `T`'s in `<control-string>`.

The activation function between two layers are always ReLU.

For example, the following command
```bash
python main.py --control_str TNTSTNTST --channels 1 16 32 32 64 128
```
specifies the following architecture:

```
+------------------------------------------------------------+
|                          Input                             |
+------------------------------------------------------------+
|  1D Conv, in_channel = 1, out_channel = 16, dilation = 1   |
+------------------------------------------------------------+
|                   Layer Normalization                      |
+------------------------------------------------------------+
|  1D Conv, in_channel = 16, out_channel = 32, dilation = 2  |
+------------------------------------------------------------+
|       Graph Conv, in_channel = 32, out_channel = 32        |
+------------------------------------------------------------+
|  1D Conv, in_channel = 32, out_channel = 32, dilation = 4  |
+------------------------------------------------------------+
|                   Layer Normalization                      |
+------------------------------------------------------------+
|  1D Conv, in_channel = 32, out_channel = 64, dilation = 8  |
+------------------------------------------------------------+
|       Graph Conv, in_channel = 64, out_channel = 64        |
+------------------------------------------------------------+
| 1D Conv, in_channel = 64, out_channel = 128, dilation = 16 |
+------------------------------------------------------------+
```

Results
-------

```bash
python main.py
```
METR_LA MAE: ~5.76
