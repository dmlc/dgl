git submodule init
git submodule update
mkdir build
cd build
cmake -DUSE_CUDA=$USE_CUDA -DUSE_OPENMP=ON -DCUDA_ARCH_NAME=All -DCUDA_TOOLKIT_ROOT_DIR=/usr/local/cuda-$CUDA_VER ..
make
cd ../python
$PYTHON setup.py install --single-version-externally-managed --record=record.txt
