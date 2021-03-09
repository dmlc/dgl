# DGL Implementation of DimeNet and DimeNet++

This DGL example implements the GNN model proposed in the paper [Directional Message Passing for Molecular Graphs](https://arxiv.org/abs/2003.03123) and [Fast and Uncertainty-Aware Directional Message Passing for Non-Equilibrium Molecules](https://arxiv.org/abs/2011.14115). For the original implementation, see [here](https://github.com/klicperajo/dimenet).

Contributor: [xnuohz](https://github.com/xnuohz)

* This example implements both DimeNet and DimeNet++.
* The advantages of DimeNet++ over DimeNet
    - Fast interactions: replacing bilinear layer with a simple Hadamard priduct
    - Embedding hierarchy: using a higher number of embeddings by reducing the embedding size in blocks via down- and up-projection layers
    - Other improvements: using less interaction blocks

### Requirements
The codebase is implemented in Python 3.6. For version requirement of packages, see below.

```
click 7.1.2
dgl 0.6.0
logzero 1.6.3
numpy 1.19.5
ruamel.yaml 0.16.12
scikit-learn 0.24.1
scipy 1.5.4
sympy 1.7.1
torch 1.7.0
tqdm 4.56.0
```

### The graph datasets used in this example

The DGL's built-in QM9 dataset. Dataset summary:

* Number of Molecular Graphs: 130,831
* Number of Tasks: 12

### Usage

**Note: DimeNet++ is recommended to use over DimeNet.**

##### Examples

The following commands learn a neural network and predict on the test set.
Training a DimeNet model on QM9 dataset.
```bash
python main.py --model-cnf config/dimenet.yaml
```
Training a DimeNet++ model on QM9 dataset.
```bash
python main.py --model-cnf config/dimenet_pp.yaml
```
For faster experimentation, you should first put the author's [pretrained](https://github.com/klicperajo/dimenet/tree/master/pretrained) folder here, which contains pre-trained TensorFlow models. You can convert a TensorFlow model to a PyTorch model by using the following commands.
```
python convert_tf_ckpt_to_pytorch.py --model-cnf config/dimenet_pp.yaml --convert-cnf config/convert.yaml
```
Then you can set `flag: True` in `dimenet_pp.yaml` and run the above script, DimeNet++ will use the pretrained weights to predict on the test set.

##### Configuration

For more details, please see `config/dimenet.yaml` and `config/dimenet_pp.yaml`

###### Model options
```
// The following paramaters are only used in DimeNet++
out_emb_size      int    Output embedding size.                                         Default is 256
int_emb_size      int    Input embedding size.                                          Default is 64
basis_emb_size    int    Basis embedding size.                                          Default is 8
extensive         bool   Readout operator for generating a graph-level representation.  Default is True 

// The following paramater is only used in DimeNet
num_bilinear      int    Third dimension of the bilinear layer tensor in DimeNet.       Default is 8

// The following paramaters are used in both DimeNet and DimeNet++
emb_size          int    Embedding size used throughout the model.                              Default is 128
num_blocks        int    Number of building blocks to be stacked.                               Default is 6 in DimeNet and 4 in DimeNet++   
num_spherical     int    Number of spherical harmonics.                                         Default is 7   
num_radial        int    Number of radial basis functions.                                      Default is 6   
envelope_exponent int    Shape of the smooth cutoff.                                            Default is 5   
cutoff            float  Cutoff distance for interatomic interactions.                          Default is 5.0 
num_before_skip   int    Number of residual layers in interaction block before skip connection. Default is 1   
num_after_skip    int    Number of residual layers in interaction block after skip connection.  Default is 2   
num_dense_output  int    Number of dense layers for the output blocks.                          Default is 3   
targets           list   List of targets to predict.                                            Default is ['mu']
output_init       string Initial function name for output layer.                                Default is 'GlorotOrthogonal'
```

###### Training options
```
num_train         int   Number of training samples.                     Default is 110000
num_valid         int   Number of validation samples.                   Default is 10000
data_seed         int   Random seed.                                    Default is 42
lr                float Learning rate.                                  Default is 0.001
weight_decay      float Weight decay.                                   Default is 0.0001
ema_decay         float EMA decay.                                      Default is 0.
batch_size        int   Batch size.                                     Default is 100
epochs            int   Training epochs.                                Default is 300
early_stopping    int   Patient epochs to wait before early stopping.   Default is 20
num_workers       int   Number of subprocesses to use for data loading. Default is 18
gpu               int   GPU index.                                      Default is 0, using CUDA:0
interval          int   Time intervals for model evaluation.            Default is 50
step_size         int   Period of learning rate decay.                  Default is 100
gamma             float Factor of learning rate decay.                  Default is 0.3
```

### Performance

- Batch size is different
- Linear learning rate warm-up is not used
- Exponential learning rate decay is not used
- Exponential moving average (EMA) is not used
- The values for tasks except mu, alpha, r2, Cv should be x 10^-3
- The author's code didn't provide the pretrained model for gap task
- MAE(DimeNet in Table 1) is from [here](https://arxiv.org/abs/2003.03123)
- MAE(DimeNet++ in Table 2) is from [here](https://arxiv.org/abs/2011.14115)

| Target | mu | alpha | homo | lumo | gap | r2 | zpve | U0 | U | H | G | Cv |
| :-: | :-: | :-: | :-: | :-: | :-: | :-: | :-: | :-: | :-: | :-: | :-: | :-: |
| MAE(DimeNet in Table 1)      | 0.0286 | 0.0469 | 27.8 | 19.7 | 34.8 | 0.331 | 1.29 | 8.02 | 7.89 | 8.11 | 8.98 | 0.0249 |
| MAE(DimeNet++ in Table 2)    | 0.0297 | 0.0435 | 24.6 | 19.5 | 32.6 | 0.331 | 1.21 | 6.32 | 6.28 | 6.53 | 7.56 | 0.0230 |
| MAE(DimeNet++, TF, pretrain) | 0.0297 | 0.0435 | 0.0246 | 0.0195 | -      | 0.3312 | 0.00121 | 0.0063 | 0.00628 | 0.00653 | 0.00756 | 0.0230 |
| MAE(DimeNet++, TF, scratch)  | 0.0330 | 0.0447 | 0.0251 | 0.0227 | 0.0486 | 0.3574 | 0.00123 | 0.0065 | 0.00635 | 0.00658 | 0.00747 | 0.0224 |
| MAE(DimeNet++, DGL)          | 0.0326 | 0.0537 | 0.0311 | 0.0255 | 0.0490 | 0.4801 | 0.0043 | 0.0141 | 0.0109 | 0.0117 | 0.0150 | 0.0254 |

### Speed

| Model | Original Implementation | DGL Implementation | Improvement |
| :-: | :-: | :-: | :-: |
| DimeNet | 2839 | 1345 | 2.1x |
| DimeNet++ | 624 | 238 | 2.6x |
