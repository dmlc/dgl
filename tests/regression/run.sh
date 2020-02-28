. /opt/conda/etc/profile.d/conda.sh

cd ~
cd regression
cd dgl
# git clone --recursive https://github.com/dmlc/dgl.git 
git pull 
git submodule init
git submodule update --recursive

conda activate base
pip install asv

for backend in pytorch mxnet tensorflow
do 
conda activate "${backend}-ci"
pip uninstall -y dgl-cu101
pip install --pre dgl-cu101
done

conda activate base
asv machine --yes
asv run
asv publish
