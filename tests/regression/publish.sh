docker run -d --name dgl-reg --rm --runtime=nvidia -v dgl-regression:/root/regression dgllib/dgl-ci-gpu:conda
docker cp ./tests/regression/run.sh dgl-red:/root/regression_run.sh
docker exec -d bash /root/regression_run.sh
docker cp /root/regression/dgl/html ~/reg_prod/html/