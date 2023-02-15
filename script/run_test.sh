#!/bin/bash

set -e

usage() {
cat << EOF
usage: bash $0 OPTIONS
examples:
  Clean and restart a CPU only build: bash $0 -c
  Clean and restart a CUDA build: bash $0 -g
  Build incrementally: bash $0

Build DGL. By default, build incrementally on top of the current state.

OPTIONS:
  -h           Show this message.
  -c           Clean and restart CPU only build.
  -g           Clean and restart CUDA build.
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
shift $(($OPTIND-1))

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

export DGLBACKEND=pytorch
export DGL_LIBRARY_PATH=${DGL_HOME}/build
export PYTHONPATH=${DGL_HOME}/python:${DGL_HOME}/tests:$PYTHONPATH
export DGLTESTDEV=${device}
export DGL_DOWNLOAD_DIR=${DGL_HOME}/build

python3 -m pytest -v $@
