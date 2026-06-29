rm -rf build
rm -rf tensoradapter/pytorch/build
rm -rf graphbolt/build
rm -rf dgl_sparse/build

bash script/build_dgl_ascend.sh
cd python
pip install -e .