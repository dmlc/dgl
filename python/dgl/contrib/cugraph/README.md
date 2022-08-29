Notes for setting up the environment

0. Do not activate cugraph-ops's conda environment
1. Manually build `dgl` from source:
```bash
mkdir build
cd build
cmake -DUSE_CUDA=ON ..
make -j
```
2. Activate the conda environment, install `pylibcugraphops`, then install `dgl` in an editable mode:
```bash
conda activate env_name
cd ../python
pip install -e .
```

The key is to compile `dgl` C layer using system nvcc and gcc.