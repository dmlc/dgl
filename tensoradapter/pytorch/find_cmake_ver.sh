#!/bin/bash
set -e

if [ $2 ]; then
	# tested on 4.5.11 ($_CONDA_PREFIX/bin/conda) and 4.8.5 ($_CONDA_PREFIX/condabin/conda)
	if [ $CONDA_EXE ]; then
		_CONDA_PREFIX=$(dirname $(dirname $CONDA_EXE))
		. $_CONDA_PREFIX/etc/profile.d/conda.sh
	else
		echo "Environment variable CONDA_EXE not set so cannot find conda"
		exit 1
	fi
	eval "$(conda shell.bash hook)"
	conda activate $2
fi
python $1
if [ $2 ]; then
	conda deactivate
fi
