docker run --name dgl-reg --rm --hostname=reg-machine --runtime=nvidia \
    -v /home/ubuntu/asv_workspace:/root/regression \
    -it dgllib/dgl-ci-gpu:conda \
    /bin/bash /root/regression/dgl/tests/regression/run.sh