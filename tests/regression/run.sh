. /opt/conda/etc/profile.d/conda.sh

cd ~
mkdir regression
cd regression
# git config core.filemode false
git clone --recursive https://github.com/dmlc/dgl.git 
cd dgl
mkdir asv
cp -r ~/asv_data/* asv/

conda activate base
pip install --upgrade pip
pip install asv numpy

source /root/regression/dgl/tests/scripts/build_dgl.sh gpu

conda activate base
asv machine --yes
asv run
asv publish
