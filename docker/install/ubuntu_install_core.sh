# install libraries for building c++ core on ubuntu
apt update && apt install -y --no-install-recommends --force-yes \
        apt-utils git build-essential make wget unzip sudo \
        libz-dev libxml2-dev libopenblas-dev libopencv-dev \
        graphviz graphviz-dev libgraphviz-dev ca-certificates \
        systemd vim openssh-client openssh-server
