#!/bin/bash

python -m grpc_tools.protoc -I. --python_out=. --grpc_python_out=. ./sampler.proto