#!/bin/bash
# Adapted from github.com/dmlc/dgl/tests/scripts/task_lint.sh

# pylint
echo 'Checking code style of python codes...'
python3 -m pylint --reports=y -v --rcfile=tests/lint/pylintrc python/dgllife || exit 1