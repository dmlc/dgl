## Introduction

DGL-Ascend enables Deep Graph Library ([DGL](https://github.com/dmlc/dgl)) to run on Ascend NPUs. It is developed by the BUPT-GAMMA group.

Before proceeding with the installation, ensure that CANN has been installed on your Ascend device.You can follow the instructions in [CANN Installation](https://ascend.github.io/docs/sources/ascend/quick_install.html) to install CANN.

## Installation

Create and activate a conda environment
```bash
conda create -n dgl-ascend python=3.10
conda activate dgl-ascend
```

Install PyTorch and torch_npu
```bash
pip install torch==2.8.0 torchvision==0.23.0 torchaudio==2.8.0 --index-url https://download.pytorch.org/whl/cpu
pip install torch_npu==2.8.0
```

Install DGL-Ascend from source

Download the source files from GitHub.
```bash
git clone https://gitcode.com/AI4Science/dgl-ascend.git
```

Update submodules
```bash
cd dgl-ascend
git submodule update --init --recursive
```

Build and compile DGL-Ascend
```bash
bash ./script/build_dgl_ascend.sh
```

Install the Python binding
```bash
cd ./python
python setup.py install
# Build Cython extension
python setup.py build_ext --inplace
```

## Quick Start example

You can use [LightGCN](examples/pytorch/lightgcn/README.md) as an example to run DGL-Ascend on Ascend NPUs.