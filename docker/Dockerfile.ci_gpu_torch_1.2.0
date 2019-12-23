# CI docker GPU env
FROM nvidia/cuda:10.1-cudnn7-devel-ubuntu16.04

RUN apt-get update --fix-missing

COPY install/ubuntu_install_core.sh /install/ubuntu_install_core.sh
RUN bash /install/ubuntu_install_core.sh

COPY install/ubuntu_install_build.sh /install/ubuntu_install_build.sh
RUN bash /install/ubuntu_install_build.sh

# python
COPY install/ubuntu_install_conda.sh /install/ubuntu_install_conda.sh
RUN bash /install/ubuntu_install_conda.sh

ENV CONDA_ALWAYS_YES="true"

COPY install/conda_env/kg_gpu.yml /install/conda_env/kg_gpu.yml
RUN ["/bin/bash", "-i", "-c", "conda env create -f /install/conda_env/kg_gpu.yml"]

ENV CONDA_ALWAYS_YES=

COPY install/FB15k.zip /data/kg/FB15k.zip
RUN cd /data/kg && unzip FB15k.zip

# Environment variables
ENV PATH=/usr/local/nvidia/bin:${PATH}
ENV PATH=/usr/local/cuda/bin:${PATH}
ENV CPLUS_INCLUDE_PATH=/usr/local/cuda/include:${CPLUS_INCLUDE_PATH}
ENV C_INCLUDE_PATH=/usr/local/cuda/include:${C_INCLUDE_PATH}
ENV LIBRARY_PATH=/usr/local/cuda/lib64:/usr/local/nvidia/lib64:${LIBRARY_PATH}
ENV LD_LIBRARY_PATH=/usr/local/cuda/lib64:/usr/local/nvidia/lib64:${LD_LIBRARY_PATH}
