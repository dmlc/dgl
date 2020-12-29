# FeatGraph-DGL

FeatGraph is an efficient backend for Graph Neural Networks based on TVM.

This folder contains the code for DGL to dynamically linking featgraph kernels compiled with TVM.

Reference: [TVM Tutorial on Deploy TVM Module using C++ API](https://tvm.apache.org/docs/deploy/cpp_deploy.html).

## Usage

Export featgraph kernels as a module.
```bash
python pack_featgraph.py
```

Verify correctness:
```bash
python test.py
```
