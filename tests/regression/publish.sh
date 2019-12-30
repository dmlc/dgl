docker run --name dgl-reg --rm --runtime=nvidia -v dgl-regression:/root/regression -d -it dgllib/dgl-ci-gpu:conda /bin/bash
docker cp ./tests/regression/run.sh dgl-reg:/root/regression_run.sh
docker exec -d dgl-reg bash /root/regression_run.sh
docker cp /root/regression/dgl/html ~/reg_prod/html/
docker stop dgl-reg