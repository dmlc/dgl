#!/bin/bash

set -e
mkdir -p sub407 && cd sub407 || exit 1
# update following line to setup gcc 8.3.0 compiler
#source /swtools/gcc/8.3.0/gcc_vars.sh
wget https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh
bash ./Miniconda3-latest-Linux-x86_64.sh -b -p ./miniconda3
miniconda3/bin/conda create -y -n sub407 python=3.7.10

echo "Activating conda env..."
source miniconda3/bin/activate sub407

echo "Installing packages...."
conda install -y numpy ninja pyyaml mkl mkl-include setuptools cmake cffi jemalloc tqdm future pydot scikit-learn
conda install -y -c intel numpy
conda install -y -c eumetsat expect
conda install -y -c conda-forge gperftools onnx tensorboardx libunwind

echo "Install pytorch..."
#conda install -y pytorch==1.7.1 torchvision==0.8.2 torchaudio==0.7.2 cpuonly -c pytorch
conda install -y pytorch==1.7.1  cpuonly -c pytorch
echo $?

echo "Install torch-ccl..."
( git clone https://github.com/ddkalamk/torch-ccl.git && cd torch-ccl && git checkout working_1.7 && git submodule sync && git submodule update --init --recursive && CMAKE_C_COMPILER=gcc CMAKE_CXX_COMPILER=g++ python setup.py install )

echo "Install pytorch..."
( git clone --recursive https://github.com/sanchit-misra/dgl.git -b xeon-optimizations && cd dgl && git checkout c4d98dd && rm -rf build && mkdir build && cd build && cmake ../ &&  make -j && cd ../python && python setup.py clean && python setup.py install ) 

echo "All installations done !!!"
