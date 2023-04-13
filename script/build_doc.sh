#!/bin/bash

set -e

usage() {
cat << EOF
usage: bash $0 OPTIONS
examples:
  Build doc with PyTorch-backend: bash $0 -p
  Build doc with MXNet-backend: bash $0 -m
  Build doc with TensorFlow-backend: bash $0 -t
  Build incrementally with PyTorch-backend: bash $0
  Remove all outputs and restart a PyTorch build: bash $0 -p -r

Build DGL documentation. By default, build incrementally on top of the current state.

OPTIONS:
  -h           Show this message.
  -p           Build doc with PyTorch backend.
  -m           Build doc with MXNet backend.
  -t           Build doc with TensorFlow backend.
  -r           Remove all outputs.
EOF
}

backend="pytorch"

# Parse flags.
while getopts "hpmtr" flag; do
  if [[ ${flag} == "p" ]]; then
    backend="pytorch"
  elif [[ ${flag} == "m" ]]; then
    backend="mxnet"
  elif [[ ${flag} == "t" ]]; then
    backend="tensorflow"
  elif [[ ${flag} == "r" ]]; then
    remove="YES"
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

cd ${DGL_HOME}/docs

if [[ ${remove} == "YES" ]]; then
  bash clean.sh
fi

export DGLBACKEND=$backend
export DGL_LIBRARY_PATH=${DGL_HOME}/build
export PYTHONPATH=${DGL_HOME}/python:$PYTHONPATH

make $backend

exit 0
