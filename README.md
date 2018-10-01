# Deep Graph Library
[![Build Status](http://216.165.71.225:8080/buildStatus/icon?job=DGL/master)](http://216.165.71.225:8080/job/DGL/job/master/)
[![GitHub license](https://dmlc.github.io/img/apache2.svg)](./LICENSE)

## Architecture
Show below, there are three sets of APIs for different models.
- `update_all`, `proppagate` are more global
- `update_by_edge`, `update_to` and `update_from` give finer control when updates are applied to a path, or a group of nodes
- `sendto` and `recvfrom` are the bottom primitives that update a message and node.

![Screenshot](graph-api.png)

## For Model developers
- Always choose the API at the *highest* possible level.
- Refer to [the default modules](examples/pytorch/util.py) to see how to register message and node update functions as well as readout functions; note how you can control sharing of parameters by adding a counter.

## How to build (the `cpp` branch)

Before building, make sure that the submodules are cloned.  If you haven't initialized the submodules, run

```sh
$ git submodule init
```

To sync the submodules, run

```sh
$ git submodule update
```

At the root directory of the repo:

```sh
$ mkdir build
$ cd build
$ cmake ..
$ make
$ export DGL_LIBRARY_PATH=$PWD
```

The `DGL_LIBRARY_PATH` environment variable should point to the library `libdgl.so` built by CMake.
