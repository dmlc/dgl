Unit test
===

## Python Unittest
The code organization goes as follows:

* `backend`: Additional unified tensor interface for supported frameworks.
  The functions there are only used in unit tests, not DGL itself.  Note that
  the code there are not unit tests by themselves.
* `compute`: All framework-agnostic computation-related unit tests go there.
* `${DGLBACKEND}` (e.g. `pytorch` and `mxnet`): All framework-specific
  computation-related unit tests go there.
* `graph_index`: All unit tests for C++ graph structure implementation go
  there.  The Python API being tested in this directory, if any, should be
  as minimal as possible (usually simple wrappers of corresponding C++
  functions).
* `lint`: Pylint-related files.
* `scripts`: Automated test scripts for CI.

## C++ Unittest
Compile with unittest by executing the command below
```
# Assume current directory is the root directory of dgl, and googletest submodule is initialized
bash script/build_dgl.sh -c -r
./runUnitTests
```
