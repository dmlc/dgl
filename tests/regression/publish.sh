docker run --name dgl-reg --rm --hostname=reg-machine --runtime=nvidia -v dgl-regression:/root/regression -d -it dgllib/dgl-ci-gpu:conda /bin/bash
docker cp ./tests/regression/run.sh dgl-reg:/root/regression_run.sh
docker exec dgl-reg bash /root/regression_run.sh
docker cp dgl-reg:/root/regression/dgl/html /home/ubuntu/reg_prod/html/
docker stop dgl-reg