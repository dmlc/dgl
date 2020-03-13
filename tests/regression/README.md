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

```bash
docker run --name dgl-reg --rm --hostname=reg-machine --runtime=nvidia -dit dgllib/dgl-ci-gpu:conda /bin/bash
docker exec dgl-reg bash /root/asv_data/run.sh
docker cp dgl-reg:/root/regression/dgl/asv/. /home/ubuntu/asv_data/  # Change /home/ubuntu/asv to the path you want to put the result
docker stop dgl-reg
```

And in the directory you choose (such as `/home/ubuntu/asv_data`), there's a `html` directory. You can use `python -m http.server` to start a server to see the result
