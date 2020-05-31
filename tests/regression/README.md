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
bash ./publish.sh <repo> <branch>
```

The running result will be at `./asv_data/`. You can use `python -m http.server` inside the `html` folder to start a server to see the result
