git clone --recursive -b dgl_graph https://github.com/zheng-da/incubator-mxnet.git
cd incubator-mxnet
make -j4 USE_OPENCV=1 USE_BLAS=openblas USE_CUDA=1 USE_CUDA_PATH=/usr/local/cuda \
  USE_CUDNN=1 USE_MKLDNN=1 USE_DIST_KVSTORE=1
pip3 install -e python
