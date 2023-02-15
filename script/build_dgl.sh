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
    cuda="OFF"
  elif [[ ${flag} == "g" ]]; then
    cuda="ON"
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

if [[ -z ${cuda} ]]; then
  if [[ -d build ]]; then
    cd build
  else
    echo "ERROR: No existing build status found, unable to build incrementally."
    usage
    exit 1
  fi
else
  rm -rf build
  mkdir -p build
  cd build
  cmake -DUSE_CUDA=${cuda} ..
fi

if [[ ${PWD} == "${DGL_HOME}/build" ]]; then
  make -j
else
  echo "ERROR: unexpected working directory."
  echo " Current: ${PWD}"
  echo "Expected: ${DGL_HOME}/build"
fi
exit 0
