# FeatGraph-DGL

FeatGraph is an efficient backend for Graph Neural Networks based on TVM.

- Original repo: https://github.com/amazon-research/FeatGraph
- SC2020 Paper: https://www.csl.cornell.edu/~zhiruz/pdfs/featgraph-sc2020.pdf 

This folder contains the code for integrating featgraph kernels to DGL.

## Usage

After building DGL with `USE_TVM=ON`, you should be able to run:
```bash
python test.py
```
to verify correctness.

## Reference

- [TVM Tutorial on Deploy TVM Module using C++ API](https://tvm.apache.org/docs/deploy/cpp_deploy.html).

