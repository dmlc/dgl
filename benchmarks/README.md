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

Note that local run will not produce any benchmark results on disk.
To change the device for benchmarking, set the `DGL_BENCH_DEVICE` environment variable.
Any valid PyTorch device strings are allowed.

```bash
export DGL_BENCH_DEVICE=cuda:0
```

DGL runs all benchmarks automatically in docker container. To run all benchmarks in docker,
use the `publish.sh` script. It accepts two arguments, a name specifying the identity of
the test machine and a device name.

```bash
bash publish.sh dev-machine cuda:0
```

The script will output two folders `results` and `html`. The `html` folder contains the
generated static web pages. View it by:

```bash
asv preview
```


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
* Feed flags `-e --verbose` to `asv run` to print out stderr and more information. Use `--bench` flag
  to run specific benchmarks.
* When running benchmarks locally (e.g., with `--python=same`), ASV will not write results to disk
  so `asv publish` will not generate plots.
* When running benchmarks in docker, ASV will pull the codes from remote and build them in conda
  environment. The repository to pull is determined by `origin`, so it works with forked repository.
  The branches are configured in `asv.conf.json`. If you wish to test the impact of your local source
  code changes on performance in docker, remember to before running `publish.sh`:
    - Commit your local changes and push it to remote `origin`.
    - Add the corresponding branch to `asv.conf.json`.
* Try make your benchmarks compatible with all the versions being tested.
