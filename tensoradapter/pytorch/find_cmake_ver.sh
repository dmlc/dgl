#!/bin/bash
set -e

if [ $2 ]; then
	if [ ! $CONDA_EXE ]; then
		echo "Cannot find CONDA_EXE"
		exit 1
	fi
	eval "$($CONDA_EXE shell.bash hook)"
	$CONDA_EXE activate $2
fi
python $1
if [ $2 ]; then
	$CONDA_EXE deactivate
fi
