# PyTorch CapsNet: Capsule Network for PyTorch

[![license](https://img.shields.io/github/license/mashape/apistatus.svg?maxAge=2592000)](https://github.com/cedrickchee/capsule-net-pytorch/blob/master/LICENSE)
![completion](https://img.shields.io/badge/completion%20state-95%25-green.svg?style=plastic)

A CUDA-enabled PyTorch implementation of CapsNet (Capsule Network) based on this paper:
[Sara Sabour, Nicholas Frosst, Geoffrey E Hinton. Dynamic Routing Between Capsules. NIPS 2017](https://arxiv.org/abs/1710.09829)

The current `test error is 0.21%` and the `best test error is 0.20%`. The current `test accuracy is 99.31%` and the `best test accuracy is 99.32%`.

**What is a Capsule**

> A Capsule is a group of neurons whose activity vector represents the instantiation parameters of a specific type of entity such as an object or object part.

You can learn more about Capsule Networks [here](#learning-resources).

**Why another CapsNet implementation?**

I wanted a decent PyTorch implementation of CapsNet and I couldn't find one at the point when I started. The goal of this implementation is focus to help newcomers learn and understand the CapsNet architecture and the idea of Capsules. The implementation is **NOT** focus on rigorous correctness of the results. In addition, the codes are not optimized for speed. To help us read and understand the codes easier, the codes comes with ample comments and the Python classes and functions are documented with Python docstring.

I will try my best to check and fix issues reported. Contributions are highly welcomed. If you find any bugs or errors in the codes, please do not hesitate to open an issue or a pull request. Thank you.

**Status and Latest Updates:**

See the [CHANGELOG](CHANGELOG.md)

**Datasets**

The model was trained on the standard [MNIST](http://yann.lecun.com/exdb/mnist/) data.

*Note: you don't have to manually download, preprocess, and load the MNIST dataset as [TorchVision](https://github.com/pytorch/vision) will take care of this step for you.*

I have tried using other datasets. See the [Other Datasets](#other-datasets) section below for more details.

## Requirements
- Python 3
  - Tested with version 3.6.4
- [PyTorch](http://pytorch.org/)
    - Tested with version 0.3.0.post4
    - Migrate existing code to work in version 0.4.0. [Work-In-Progress]
    - Code will not run with version 0.1.2 due to `keepdim` not available in this version.
    - Code will not run with version 0.2.0 due to `softmax` function doesn't takes a dimension.
- CUDA 8 and above
  - Tested with CUDA 8 and CUDA 9.
- [TorchVision](https://github.com/pytorch/vision)
- [tensorboardX](https://github.com/lanpa/tensorboard-pytorch)
- [tqdm](https://github.com/tqdm/tqdm)

## Usage

### Training and Evaluation
**Step 1.**
Clone this repository with ``git`` and install project dependencies.

```bash
$ git clone https://github.com/cedrickchee/capsule-net-pytorch.git
$ cd capsule-net-pytorch
$ pip install -r requirements.txt
```

**Step 2.** 
Start the CapsNet on MNIST training and evaluation:

- Training with default settings:
```bash
$ python main.py
```

- Training on 8 GPUs with 30 epochs and 1 routing iteration:
```bash
$ CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 python main.py --epochs 30 --num-routing 1 --threads 16 --batch-size 128 --test-batch-size 128
```

**Step 3.**
Test a pre-trained model:

If you have trained a model in Step 2 above, then the weights for the trained model will be saved to `results/trained_model/model_epoch_10.pth`. [WIP] Now just run the following command to get test results.

```bash
$ python main.py --is-training 0 --weights results/trained_model/model_epoch_10.pth
```

### Pre-trained Model

You can download the weights for the pre-trained model from my Google Drive. We saved the weights (model state dict) and the optimizer state for the model at the end of every training epoch.

- Weights from [epoch 50 checkpoint](https://drive.google.com/uc?export=download&id=1lYtOMSreP4I9hr9un4DsBJZrzodI6l2d) [84 MB].
- Weights from [epoch 40 to 50](https://drive.google.com/uc?export=download&id=1VMuVtJrecz47czsT5HqLxZpFjkLoMKaL) checkpoints [928 MB].

Uncompress and put the weights (.pth files) into `./results/trained_model/`.

*Note: the model was **last trained** on 2017-11-26 and the weights **last updated** on 2017-11-28.*

### The Default Hyper Parameters

| Parameter | Value | CLI arguments |
| --- | --- | --- |
| Training epochs | 10 | --epochs 10 |
| Learning rate | 0.01 | --lr 0.01 |
| Training batch size | 128 | --batch-size 128 |
| Testing batch size | 128 | --test-batch-size 128 |
| Log interval | 10 | --log-interval 10 |
| Disables CUDA training | false | --no-cuda |
| Num. of channels produced by the convolution | 256 | --num-conv-out-channel 256 |
| Num. of input channels to the convolution | 1 | --num-conv-in-channel 1 |
| Num. of primary unit | 8 | --num-primary-unit 8 |
| Primary unit size | 1152 | --primary-unit-size 1152 |
| Num. of digit classes | 10 | --num-classes 10 |
| Output unit size | 16 | --output-unit-size 16 |
| Num. routing iteration | 3 | --num-routing 3 |
| Use reconstruction loss | true | --use-reconstruction-loss |
| Regularization coefficient for reconstruction loss | 0.0005 | --regularization-scale 0.0005 |
| Dataset name (mnist, cifar10) | mnist | --dataset mnist |
| Input image width to the convolution | 28 | --input-width 28 |
| Input image height to the convolution | 28 | --input-height 28 |

## Results

### Test Error

CapsNet classification test error on MNIST. The MNIST average and standard deviation results are reported from 3 trials.

The results can be reproduced by running the following commands.

```bash
 python main.py --epochs 50 --num-routing 1 --use-reconstruction-loss no --regularization-scale 0.0       #CapsNet-v1
 python main.py --epochs 50 --num-routing 1 --use-reconstruction-loss yes --regularization-scale 0.0005   #CapsNet-v2
 python main.py --epochs 50 --num-routing 3 --use-reconstruction-loss no --regularization-scale 0.0       #CapsNet-v3
 python main.py --epochs 50 --num-routing 3 --use-reconstruction-loss yes --regularization-scale 0.0005   #CapsNet-v4
```

Method | Routing | Reconstruction | MNIST (%) | *Paper*
:---------|:------:|:---:|:----:|:----:
Baseline |  -- | -- | -- | *0.39*
CapsNet-v1 | 1 | no | -- | *0.34 (0.032)*
CapsNet-v2 | 1 | yes | -- | *0.29 (0.011)*
CapsNet-v3 | 3 | no | -- | *0.35 (0.036)*
CapsNet-v4 | 3 | yes | 0.21 | *0.25 (0.005)*

### Training Loss and Accuracy

The training losses and accuracies for CapsNet-v4 (50 epochs, 3 routing iteration, using reconstruction, regularization scale of 0.0005):

![](results/train_loss_accuracy.png)

Training accuracy. Highest training accuracy: 100%

![](results/train_accuracy.png)

Training loss. Lowest training error: 0.1938%

![](results/train_loss.png)

### Test Loss and Accuracy

The test losses and accuracies for CapsNet-v4 (50 epochs, 3 routing iteration, using reconstruction, regularization scale of 0.0005):

![](results/test_loss_accuracy.png)

Test accuracy. Highest test accuracy: 99.32%

![](results/test_accuracy.png)

Test loss. Lowest test error: 0.2002%

![](results/test_loss.png)

### Training Speed

- Around `5.97s / batch` or `8min / epoch` on a single Tesla K80 GPU with batch size of 704.
- Around `3.25s / batch` or `25min / epoch` on a single Tesla K80 GPUwith batch size of 128.

![](results/training_speed.png)

In my case, these are the hyperparameters I used for the training setup:

- batch size: 128
- Epochs: 50
- Num. of routing: 3
- Use reconstruction loss: yes
- Regularization scale for reconstruction loss: 0.0005

### Reconstruction

The results of CapsNet-v4.

Digits at left are reconstructed images.
<table>
  <tr>
    <td>
     <img src="results/reconstructed_images.png"/>
    </td>
    <td>
      <p>[WIP] Ground truth image from dataset</p>
    </td>
  </tr>
</table>

### Model Design

```bash
Model architecture:
------------------

Net (
  (conv1): ConvLayer (
    (conv0): Conv2d(1, 256, kernel_size=(9, 9), stride=(1, 1))
    (relu): ReLU (inplace)
  )
  (primary): CapsuleLayer (
    (conv_units): ModuleList (
      (0): Conv2d(256, 32, kernel_size=(9, 9), stride=(2, 2))
      (1): Conv2d(256, 32, kernel_size=(9, 9), stride=(2, 2))
      (2): Conv2d(256, 32, kernel_size=(9, 9), stride=(2, 2))
      (3): Conv2d(256, 32, kernel_size=(9, 9), stride=(2, 2))
      (4): Conv2d(256, 32, kernel_size=(9, 9), stride=(2, 2))
      (5): Conv2d(256, 32, kernel_size=(9, 9), stride=(2, 2))
      (6): Conv2d(256, 32, kernel_size=(9, 9), stride=(2, 2))
      (7): Conv2d(256, 32, kernel_size=(9, 9), stride=(2, 2))
    )
  )
  (digits): CapsuleLayer (
  )
  (decoder): Decoder (
    (fc1): Linear (160 -> 512)
    (fc2): Linear (512 -> 1024)
    (fc3): Linear (1024 -> 784)
    (relu): ReLU (inplace)
    (sigmoid): Sigmoid ()
  )
)

Parameters and size:
-------------------

conv1.conv0.weight: [256, 1, 9, 9]
conv1.conv0.bias: [256]
primary.conv_units.0.weight: [32, 256, 9, 9]
primary.conv_units.0.bias: [32]
primary.conv_units.1.weight: [32, 256, 9, 9]
primary.conv_units.1.bias: [32]
primary.conv_units.2.weight: [32, 256, 9, 9]
primary.conv_units.2.bias: [32]
primary.conv_units.3.weight: [32, 256, 9, 9]
primary.conv_units.3.bias: [32]
primary.conv_units.4.weight: [32, 256, 9, 9]
primary.conv_units.4.bias: [32]
primary.conv_units.5.weight: [32, 256, 9, 9]
primary.conv_units.5.bias: [32]
primary.conv_units.6.weight: [32, 256, 9, 9]
primary.conv_units.6.bias: [32]
primary.conv_units.7.weight: [32, 256, 9, 9]
primary.conv_units.7.bias: [32]
digits.weight: [1, 1152, 10, 16, 8]
decoder.fc1.weight: [512, 160]
decoder.fc1.bias: [512]
decoder.fc2.weight: [1024, 512]
decoder.fc2.bias: [1024]
decoder.fc3.weight: [784, 1024]
decoder.fc3.bias: [784]

Total number of parameters on (with reconstruction network): 8227088 (8 million)
```

### TensorBoard

We logged the training and test losses and accuracies using tensorboardX. TensorBoard helps us visualize how the machine learn over time. We can visualize statistics, such as how the objective function is changing or weights or accuracy varied during training.

TensorBoard operates by reading TensorFlow data (events files).

#### How to Use TensorBoard

1. Download a [copy of the events files](https://drive.google.com/uc?export=download&id=1lZVffeZTkUQfSxmZmYDViRzmhb59wBWL) for the latest run from my Google Drive.
2. Uncompress the file and put it into `./runs`.
3. Check to ensure you have installed tensorflow (CPU version). We need this for TensorBoard server and dashboard.
4. Start TensorBoard.
```bash
$ tensorboard --logdir runs
```
5. Open TensorBoard dashboard in your web browser using this URL: http://localhost:6006

### Other Datasets

#### CIFAR10

In the spirit of experiment, I have tried using other datasets. I have updated the implementation so that it supports and works with CIFAR10. Need to note that I have not tested throughly our capsule model on CIFAR10.

Here's how we can train and test the model on CIFAR10 by running the following commands.

```bash
python main.py --dataset cifar10 --num-conv-in-channel 3 --input-width 32 --input-height 32 --primary-unit-size 2048 --epochs 80 --num-routing 1 --use-reconstruction-loss yes --regularization-scale 0.0005
```

##### Training Loss and Accuracy

The training losses and accuracies for CapsNet-v4 (80 epochs, 3 routing iteration, using reconstruction, regularization scale of 0.0005):

![](results/cifar10/train_loss_accuracy.png)

- Highest training accuracy: 100%
- Lowest training error: 0.3589%

##### Test Loss and Accuracy

The test losses and accuracies for CapsNet-v4 (80 epochs, 3 routing iteration, using reconstruction, regularization scale of 0.0005):

![](results/cifar10/test_loss_accuracy.png)

- Highest test accuracy: 71%
- Lowest test error: 0.5735%

## TODO
- [x] Publish results.
- [x] More testing.
- [ ] Inference mode - command to test a pre-trained model.
- [ ] Jupyter Notebook version.
- [x] Create a sample to show how we can apply CapsNet to real-world application.
- [ ] Experiment with CapsNet:
    * [x] Try using another dataset.
    * [ ] Come out a more creative model structure.
- [x] Pre-trained model and weights.
- [x] Add visualization for training and evaluation metrics.
- [x] Implement recontruction loss.
- [x] Check algorithm for correctness.
- [x] Update results from TensorBoard after making improvements and bug fixes.
- [x] Publish updated pre-trained model weights.
- [x] Log the original and reconstructed images using TensorBoard.
- [ ] Update results with reconstructed image and original image.
- [ ] Resume training by loading model checkpoint.
- [ ] Migrate existing code to work in PyTorch 0.4.0.

*WIP is an acronym for Work-In-Progress*

## Credits

Referenced these implementations mainly for sanity check:
1. [TensorFlow implementation by @naturomics](https://github.com/naturomics/CapsNet-Tensorflow)

## Learning Resources

Here's some resources that we think will be helpful if you want to learn more about Capsule Networks:

- Articles and blog posts:
  - [Understanding Hinton's Capsule Networks. Part I: Intuition.](https://medium.com/@pechyonkin/understanding-hintons-capsule-networks-part-i-intuition-b4b559d1159b)
  - [Dynamic routing between capsules](https://blog.acolyer.org/2017/11/13/dynamic-routing-between-capsules/)
  - [What is a CapsNet or Capsule Network?](https://hackernoon.com/what-is-a-capsnet-or-capsule-network-2bfbe48769cc)
  - [Capsule Networks Are Shaking up AI — Here's How to Use Them](https://hackernoon.com/capsule-networks-are-shaking-up-ai-heres-how-to-use-them-c233a0971952)
  - [Capsule Networks Explained](https://kndrck.co/posts/capsule_networks_explained/)
- Videos:
  - [Capsule Networks: An Improvement to Convolutional Networks](https://www.youtube.com/watch?v=VKoLGnq15RM)
  - [Capsule Networks (CapsNets) – Tutorial](https://www.youtube.com/watch?v=pPN8d0E3900)

## Other Implementations

- TensorFlow:
  - The first author of the paper, [Sara Sabour has released the code](https://github.com/Sarasra/models/tree/master/research/capsules).

## Real-world Application of CapsNet

The following is a few samples in the wild that show how we can apply CapsNet to real-world use cases.

- [An attempt to implement CapsNet for car make-model classification](https://www.reddit.com/r/MachineLearning/comments/80eiz3/p_implementing_a_capsnet_for_car_makemodel/)
- [A Keras implementation of Capsule Network on Fashion MNIST dataset](https://github.com/XifengGuo/CapsNet-Fashion-MNIST)