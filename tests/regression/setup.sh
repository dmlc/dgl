. /opt/conda/etc/profile.d/conda.sh

cd ~
mkdir regression
cd regression
git clone --recursive https://github.com/dmlc/dgl.git 

conda activate base
pip install asv
asv machine --config tests/regression/.asv-machine.json

asv run -E "conda:pytorch-ci"