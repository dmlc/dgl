DGL Benchmarks
====

Benchmarking DGL with Airspeed Velocity.

Usage
---

Before beginning, ensure that airspeed velocity is installed:

```bash
pip install asv
```

To run all benchmarks locally, build the project first and then run:

```bash
asv run -n -e --python=same --verbose
```

**Due to ASV's restriction, `--python=same` will not write any benchmark results
to disk. It does not support specifying branches and commits either. They are only
available under ASV's managed environment.**

To change the device for benchmarking, set the `DGL_BENCH_DEVICE` environment variable.
Allowed values are `"cpu"` or `"gpu"`.

```bash
export DGL_BENCH_DEVICE=gpu
```

To select which benchmark to run, use the `--bench` flag. For example,

```bash
asv run -n -e --python=same --verbose --bench model_acc.bench_gat
```

Note that OGB dataset need to be download manually to `/tmp/dataset` folder (i.e. `/tmp/dataset/ogbn-products/`) beforehand. 
You can do it by runnnig the code below in this folder
```python
from benchmarks.utils import get_ogb_graph
get_ogb_graph("ogbn-product")
```

Run in docker locally
---

DGL runs all benchmarks automatically in docker container. To run bencmarks in docker locally,

* Git commit your locally changes. No need to push to remote repository.
* To compare commits from different branches. Change the `"branches"` list in `asv.conf.json`.
  The default is `"HEAD"` which is the last commit of the current branch. For example, to
  compare your proposed changes with the master branch, set it to be `["HEAD", "master"]`.
  If your workspace is a forked repository, make sure your local master has synced with
  the upstream.
* Use the `publish.sh` script. It accepts two arguments, a name specifying the identity of
  the test machine and a device name. For example,
  ```bash
  bash publish.sh dev-machine gpu
  ```

The script will output two folders `results` and `html`. The `html` folder contains the
generated static web pages. View it by:

```bash
asv preview
```

Please see `publish.sh` for more information on how it works and how to modify it according
to your need.

Adding a new benchmark suite
---

The benchmark folder is organized as follows:

```
|-- benchmarks/
  |-- model_acc/           # benchmarks for model accuracy
    |-- bench_gcn.py
    |-- bench_gat.py
    |-- bench_sage.py
    ...
  |-- model_speed/         # benchmarks for model training speed
    |-- bench_gat.py
    |-- bench_sage.py
    ...
  ...                      # other types of benchmarks
|-- html/                  # generated html files
|-- results/               # generated result files
|-- asv.conf.json          # asv config file
|-- build_dgl_asv.sh       # script for building dgl in asv
|-- install_dgl_asv.sh     # script for installing dgl in asv
|-- publish.sh             # script for running benchmarks in docker
|-- README.md              # this readme
|-- run.sh                 # script for calling asv in docker
|-- ...                    # other aux files
```

To add a new benchmark, pick a suitable benchmark type and create a python script under
it. We prefer to have the prefix `bench_` in the name. Here is a toy example:

```python
# bench_range.py

import time
from .. import utils

@utils.benchmark('time')
@utils.parametrize('l', [10, 100, 1000])
@utils.parametrize('u', [10, 100, 1000])
def track_time(l, u):
    t0 = time.time()
    for i in range(l, u):
        pass
    return time.time() - t0
```

* The main entry point of each benchmark script is a `track_*` function. The function
  can have arbitrary arguments and must return the benchmark result.
* There are two useful decorators: `utils.benchmark` and `utils.parametrize`.
* `utils.benchmark` indicates the type of this benchmark. Currently supported types are:
  `'time'` and `'acc'`. The decorator will perform some necessary setup and finalize
  steps such as fixing the random seed for the `'acc'` type.
* `utils.parametrize` specifies the parameters to test.
  Multiple parametrize decorators mean benchmarking the combination.
* Check out `model_acc/bench_gcn.py` and `model_speed/bench_sage.py`.
* ASV's [official guide on writing benchmarks](https://asv.readthedocs.io/en/stable/writing_benchmarks.html)
  is also very helpful.


Tips
----
* Feed flags `-e --verbose` to `asv run` to print out stderr and more information.
* When running benchmarks locally (e.g., with `--python=same`), ASV will not write results to disk
  so `asv publish` will not generate plots.
* Try make your benchmarks compatible with all the versions being tested.
* For ogbn dataset, put the dataset into /tmp/dataset/
