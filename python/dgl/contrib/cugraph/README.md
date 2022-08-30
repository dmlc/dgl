### Notes for setting up the environment

0. Do not activate cugraph-ops's conda environment
1. Manually build `dgl` from source, following [dgl documentation](https://docs.dgl.ai/install/index.html#install-from-source):
```bash
mkdir build
cd build
cmake -DUSE_CUDA=ON ..
make -j
```
And **do not** install python bindings yet.

2. Activate the conda environment of cugraphops, install `pylibcugraphops`, then install `dgl` in an editable mode:
```bash
conda activate env_name
cd ../python
pip install -e .
```

The key is to compile `dgl` C layer using system nvcc and gcc.
