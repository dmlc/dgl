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
- Refer to the [GCN example](examples/pytorch/gcn/gcn_batch.py) to see how to register message and node update functions;

## How to build (the `cpp` branch)

Before building, make sure that the submodules are cloned.  If you haven't initialized the submodules, run

```sh
$ git submodule init
```

To sync the submodules, run

```sh
$ git submodule update
```

### Linux

At the root directory of the repo:

```sh
$ mkdir build
$ cd build
$ cmake ..
$ make
$ export DGL_LIBRARY_PATH=$PWD
```

The `DGL_LIBRARY_PATH` environment variable should point to the library `libdgl.so` built by CMake.

### Windows/MinGW (Experimental)

Make sure you have the following installed:

* CMake
* MinGW/GCC (G++)
* MinGW/Make

You can grab them from Anaconda.

In the command line prompt, run:

```
> md build
> cd build
> cmake -DCMAKE_CXX_FLAGS="-DDMLC_LOG_STACK_TRACE=0 -DTVM_EXPORTS" .. -G "MinGW Makefiles"
> mingw32-make
> set DGL_LIBRARY_PATH=%CD%
```
