. /opt/conda/etc/profile.d/conda.sh

cd ~
cd regression
rm dgl -rf
git clone --recursive https://github.com/dmlc/dgl.git 

conda activate base
pip install asv

for backend in pytorch mxnet tensorflow
do 
conda activate "${backend}-ci"
pip uninstall -y dgl-cu101
pip install --pre dgl-cu101
done

conda activate base
asv machine --config tests/regression/.asv-machine.json
asv run
asv publish
