#!/bin/bash

set -e

usage() {
cat << EOF
usage: bash $0 OPTIONS TARGETS
examples:
  Run python tests on CPU: bash $0 -c tests/compute/test_subgraph.py
  Run python tests on GPU: bash $0 -g tests/compute/test_subgraph.py

Run DGL python tests.

OPTIONS:
  -h           Show this message.
  -c           Run python tests on CPU.
  -g           Run python tests on GPU.
EOF
}

# Parse flags.
while getopts "cgh" flag; do
  if [[ ${flag} == "c" ]]; then
    device="cpu"
  elif [[ ${flag} == "g" ]]; then
    device="gpu"
  elif [[ ${flag} == "h" ]]; then
    usage
    exit 0
  else
    usage
    exit 1
  fi
done

if [[ -z ${DGL_HOME} ]]; then
  echo "ERROR: Please make sure environment variable DGL_HOME is set correctly."
  exit 1
fi

if [[ ! ${PWD} == ${DGL_HOME} ]]; then
  echo "ERROR: This script only works properly from DGL root directory."
  echo " Current: ${PWD}"
  echo "DGL_HOME: ${DGL_HOME}"
  exit 1
fi

if [[ -z ${device} ]]; then
  echo "ERROR: Test device unspecified."
  usage
  exit 1
fi

# Reset the index for non-option arguments.
shift $(($OPTIND-1))

export DGLBACKEND=pytorch
export DGL_LIBRARY_PATH=${DGL_HOME}/build
export PYTHONPATH=${DGL_HOME}/python:${DGL_HOME}/tests:$PYTHONPATH
export DGLTESTDEV=${device}
export DGL_DOWNLOAD_DIR=${DGL_HOME}/_download

if [[ -z $@ ]]; then
  echo "ERROR: Missing test targets."
  usage
  exit 1
fi

python3 -m pytest -v $@
