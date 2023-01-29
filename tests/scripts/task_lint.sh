#!/bin/bash

# cpplint
echo 'Checking code style of C++ codes...'
python3 tests/lint/lint.py dgl cpp include src || exit 1
python3 tests/lint/lint.py dgl_sparse cpp dgl_sparse/include dgl_sparse/src || exit 1

# pylint
echo 'Checking code style of python codes...'
python3 -m pylint --reports=y -v --rcfile=tests/lint/pylintrc python/dgl || exit 1
