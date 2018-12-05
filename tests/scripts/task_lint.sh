#!/bin/bash

# cpplint
echo 'Checking code style of C++ codes...'
python3 third_party/dmlc-core/scripts/lint.py dgl cpp include src
