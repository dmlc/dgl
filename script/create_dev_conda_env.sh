#!/bin/bash

readonly CUDA_VERSIONS="10.2,11.3,11.6,11.7"
readonly TORCH_VERSION="1.12.0"

usage() {
cat << EOF
usage: bash $0 OPTIONS
examples:
  bash $0 -c
  bash $0 -g 11.7

Create a developement environment for DGL developers.

OPTIONS:
  -h           Show this message.
  -c           Create dev environment in CPU mode.
  -g           Create dev environment in GPU mode with specified CUDA version,
               supported: ${CUDA_VERSIONS}.
EOF
}

validate() {
  values=$(echo "$1" | tr "," "\n")
  for value in ${values}
  do
    if [[ "${value}" == $2 ]]; then
      return 0
    fi
  done
  return 1
}

confirm() {
  echo "Continue? [yes/no]:"
  read confirm
  if [[ ! ${confirm} == "yes" ]]; then
    exit 0
  fi
}

# Parse flags.
while getopts "cg:h" flag; do
  if [[ ${flag} == "c" ]]; then
    cpu=1
  elif [[ ${flag} == "g" ]]; then
    gpu=${OPTARG}
  elif [[ ${flag} == "h" ]]; then
    usage
    exit 0
  else
    usage
    exit 1
  fi
done

if [[ -n ${gpu} && ${cpu} -eq 1 ]]; then
  echo "Only one mode can be specified."
  exit 1
fi

if [[ -z ${gpu} && -z ${cpu} ]]; then
  usage
  exit 1
fi

# Set up CPU mode.
if [[ ${cpu} -eq 1 ]]; then
  torchversion=${TORCH_VERSION}"+cpu"
  name="dgl-dev-cpu"
fi

# Set up GPU mode.
if [[ -n ${gpu} ]]; then
  if ! validate ${CUDA_VERSIONS} ${gpu}; then
    echo "Error: Invalid CUDA version."
    usage
    exit 1
  fi

  echo "Confirm the installed CUDA version matches the specified one."
  confirm

  torchversion=${TORCH_VERSION}"+cu"${gpu//[-._]/}
  name="dgl-dev-gpu"
fi

echo "Confirm you are excuting the script from your DGL root directory."
echo "Current working directory: ${PWD}"
confirm

# Prepare the conda environment yaml file.
rand=$(echo "${RANDOM}" | md5sum | head -c 20)
mkdir -p /tmp/${rand}
cp script/dgl_dev.yml.template /tmp/${rand}/dgl_dev.yml
sed -i "s|__NAME__|${name}|g" /tmp/${rand}/dgl_dev.yml
sed -i "s|__TORCH_VERSION__|${torchversion}|g" /tmp/${rand}/dgl_dev.yml
sed -i "s|__DGL_HOME__|${PWD}|g" /tmp/${rand}/dgl_dev.yml

# Ask for final confirmation.
echo "--------------------------------------------------"
cat /tmp/${rand}/dgl_dev.yml
echo "--------------------------------------------------"
echo "Create a conda enviroment with the config?"
confirm

# Create conda environment.
conda env create -f /tmp/${rand}/dgl_dev.yml

# Clean up created tmp conda environment yaml file.
rm -rf /tmp/${rand}
exit 0
