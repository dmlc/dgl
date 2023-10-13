#!/bin/bash

set -e

usage() {
cat << EOF
usage: bash $0 OPTIONS
examples:
  Start a CPU only build: bash $0 -c
  Start a CUDA build: bash $0 -g
  Build incrementally: bash $0
  Remove all intermediate output and restart a CPU only build: bash $0 -c -r
  Build with extra cmake arguments: bash $0 -c -e '-DBUILD_TORCH=ON'

Build DGL. By default, build incrementally on top of the current state.

OPTIONS:
  -h           Show this message.
  -c           Restart CPU only build.
  -e           Extra arguments of cmake.
  -g           Restart CUDA build.
  -r           Remove all intermediate output.
  -t           Type of the build: dev, dogfood or release (default: dev).
EOF
}

# Parse flags.
while getopts "ce:ghrt:" flag; do
  if [[ ${flag} == "c" ]]; then
    cuda="OFF"
  elif [[ ${flag} == "e" ]]; then
    extra_args=${OPTARG}
  elif [[ ${flag} == "g" ]]; then
    cuda="ON"
  elif [[ ${flag} == "r" ]]; then
    remove="YES"
  elif [[ ${flag} == "t" ]]; then
    build_type=${OPTARG}
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

if [[ ${remove} == "YES" ]]; then
  rm -rf build
  rm -rf graphbolt/build
  rm -rf dgl_sparse/build
  rm -rf tensoradapter/pytorch/build
fi

if [[ -z ${build_type} ]]; then
  build_type="dev"
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
  mkdir -p build
  cd build
  cmake -DBUILD_TYPE=${build_type} -DUSE_CUDA=${cuda} ${extra_args} ..
fi

if [[ ${PWD} == "${DGL_HOME}/build" ]]; then
  make -j
else
  echo "ERROR: unexpected working directory."
  echo " Current: ${PWD}"
  echo "Expected: ${DGL_HOME}/build"
fi
exit 0
