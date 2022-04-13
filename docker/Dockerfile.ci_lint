# CI docker for lint
# Adapted from github.com/dmlc/tvm/docker/Dockerfile.ci_lint

FROM ubuntu:18.04

ENV DEBIAN_FRONTEND=noninteractive

COPY install/ubuntu_install_core.sh /install/ubuntu_install_core.sh
RUN bash /install/ubuntu_install_core.sh

COPY install/ubuntu_install_python.sh /install/ubuntu_install_python.sh
RUN bash /install/ubuntu_install_python.sh

RUN apt-get install -y doxygen graphviz

RUN pip3 install cpplint==1.3.0 pylint==2.7.0 mypy
