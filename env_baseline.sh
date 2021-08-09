#!/bin/bash

source sub407_baseline/miniconda3/bin/activate sub407_baseline
#source ${CONDA_PREFIX}/lib/python3.8/site-packages/torch_ccl-1.1.0+633a77e-py3.8-linux-x86_64.egg/torch_ccl/env/setvars.sh
#torch_ccl_path=$(python -c "import torch; import torch_ccl; import os;  print(os.path.abspath(os.path.dirname(torch_ccl.__file__)))")
#source $torch_ccl_path/env/setvars.sh

NUM_THREADS=`$PREFIX lscpu | grep "Core(s) per socket" | awk '{print $NF}'`
echo "NUM_THREADS: "$NUM_THREADS
export OMP_NUM_THREADS=${NUM_THREADS}
export KMP_AFFINITY=compact,1,granularity=fine,verbose
export KMP_BLOCKTIME=1
echo "CONDA_PREFIX: "$CONDA_PREFIX
#export LD_PRELOAD=${CONDA_PREFIX}/lib/libtcmalloc.so:${CONDA_PREFIX}/lib/libiomp5.so
export LD_PRELOAD=${CONDA_PREFIX}/lib/libiomp5.so
cdir=`pwd`
export PYTHONPATH=$PYTHONPATH:$cdir/sub407_baseline/dgl/examples/pytorch/graphsage/


