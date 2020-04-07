
pushd python
for backend in pytorch mxnet tensorflow
do 
conda activate "${backend}-ci"
rm -rf build *.egg-info dist
pip uninstall -y dgl
# test install
python3 setup.py install
# test inplace build (for cython)
python3 setup.py build_ext --inplace
done
popd
conda deactivate