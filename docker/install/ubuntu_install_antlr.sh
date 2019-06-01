#!/bin/bash

set -e
set -u
set -o pipefail

cd /usr/local/lib
wget -q https://www.antlr.org/download/antlr-4.7.1-complete.jar
cd -
