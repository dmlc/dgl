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

source /root/regression/dgl/tests/scripts/build_dgl.sh gpu

conda activate base
asv machine --yes
asv run
asv publish
