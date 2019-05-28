# install cmake 3.9
wget https://cmake.org/files/v3.9/cmake-3.9.0.tar.gz
tar xvf cmake-3.9.0.tar.gz
cd cmake-3.9.0
./configure
make -j4
make install
cd ..
