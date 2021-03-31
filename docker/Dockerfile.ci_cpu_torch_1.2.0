# CI docker CPU env
# Adapted from github.com/dmlc/tvm/docker/Dockerfile.ci_cpu
FROM ubuntu:16.04

RUN apt-get update --fix-missing

COPY install/ubuntu_install_core.sh /install/ubuntu_install_core.sh
RUN bash /install/ubuntu_install_core.sh

COPY install/ubuntu_install_build.sh /install/ubuntu_install_build.sh
RUN bash /install/ubuntu_install_build.sh

# python
COPY install/ubuntu_install_conda.sh /install/ubuntu_install_conda.sh
RUN bash /install/ubuntu_install_conda.sh

ENV CONDA_ALWAYS_YES="true"

COPY install/conda_env/kg_cpu.yml /install/conda_env/kg_cpu.yml
RUN ["/bin/bash", "-i", "-c", "conda env create -f /install/conda_env/kg_cpu.yml"]

ENV CONDA_ALWAYS_YES=