git submodule init
git submodule update --recursive
mkdir build
cd build
cmake -DUSE_CUDA=$USE_CUDA -DUSE_OPENMP=ON -DCUDA_ARCH_NAME=All ..
make
cd ../python
$PYTHON setup.py install --single-version-externally-managed --record=record.txt
