How to add test to regression
=================================

Official link to [asv](https://asv.readthedocs.io/en/stable/writing_benchmarks.html)


## Add test

DGL reuses the ci docker image for the regression test. There are four conda envs, base, mxnet-ci, pytorch-ci, and tensorflow-ci.

The basic use is execute a script, and get the needed results out of the printed results.

- Create a new file in the tests/regression/
- Follow the example `bench_gcn.py` or the [official instruction](https://asv.readthedocs.io/en/stable/writing_benchmarks.html)
  - function name starts with `track` will be used to generate the stats, by the return value
  - setup function would be execute every time before running track function
  - Can use params to pass parameter into `setup` and `track_` functions

## Run locally

The default regression branch in asv is `master`. If you need to run on other branch on your fork, please change the `branches` value in the `asv.conf.json` at the root of your repo.

```bash
docker run --name dgl-reg --rm --hostname=reg-machine --runtime=nvidia -dit dgllib/dgl-ci-gpu:conda /bin/bash
docker cp ./asv_data dgl-reg:/root/asv_data/
docker cp ./run.sh dgl-reg:/root/run.sh <repo> <branch>
docker exec dgl-reg bash /root/asv_data/run.sh
docker cp dgl-reg:/root/regression/dgl/asv/. ./asv_data/  # Change /home/ubuntu/asv to the path you want to put the result
docker stop dgl-reg
```

The running result will be at `./asv_data/`. You can use `python -m http.server` inside the `html` folder to start a server to see the result
