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
pip install torch_npu==2.8.0.post2
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

## Runtime notes (important)

DGL-Ascend is a binary distribution; `libdgl.so` is linked at build time against
PyTorch (`libtorch`/`c10`) and the Ascend CANN runtime (`hccl`/`ascendcl`/`acl`,
plus `c10_npu` from `torch_npu`). Keep the following in mind at runtime:

1. **CANN must be installed and sourced first.** Ensure CANN Toolkit is present
   and its environment is loaded (`source $ASCEND_TOOLKIT_HOME/bin/setenv.bash`
   or set `LD_LIBRARY_PATH` to include `$ASCEND_TOOLKIT_HOME/lib64`), otherwise
   loading `libdgl.so` fails with `cannot open shared object file` for the CANN
   libraries.

2. **Import order matters.** `import dgl` immediately `dlopen`s `libdgl.so`,
   which must resolve `libtorch`/`c10`/`c10_npu` symbols. Always import
   `torch` and `torch_npu` **before** `dgl`:
   ```python
   import torch        # 1st
   import torch_npu    # 2nd
   import dgl          # 3rd  -- triggers ctypes.CDLL("libdgl.so")
   ```
   Importing `dgl` first will raise `ImportError: libtorch_npu.so: cannot open
   shared object file` (or similar) because the PyTorch NPU libraries have not
   been loaded into the global symbol namespace yet.

3. **Versions must match the build.** The wheel is compiled against a specific
   PyTorch C++ ABI. Install exactly the `torch`/`torch_npu` versions declared in
   `python/setup.py` `install_requires` (currently `torch==2.8.0`,
   `torch_npu==2.8.0.post2`). A mismatched pair compiles/installs but segfaults on the
   first DGL kernel call.