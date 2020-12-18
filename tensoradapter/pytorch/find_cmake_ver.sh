#!/bin/bash

if [ $2 ]; then
	eval "$(conda shell.bash hook)"
	conda activate $2
fi
python $1
if [ $2 ]; then
	conda deactivate
fi
