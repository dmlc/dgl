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
generated website hosting all the benchmark results.


Adding a new benchmark suite
---


