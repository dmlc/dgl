docker run --name dgl-reg --rm --hostname=reg-machine --runtime=nvidia -dit dgllib/dgl-ci-gpu:conda /bin/bash
docker cp /home/ubuntu/asv_data dgl-reg:/root/asv_data/
docker exec dgl-reg bash /root/asv_data/run.sh
docker cp dgl-reg:/root/regression/dgl/asv/. /home/ubuntu/asv_data/
docker stop dgl-reg