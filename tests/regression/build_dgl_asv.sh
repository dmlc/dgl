mkdir build

CMAKE_VARS="-DUSE_CUDA=ON"

rm -rf _download

pushd build
cmake $CMAKE_VARS ..
make -j4
popd
