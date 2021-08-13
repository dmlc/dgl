#!/bin/bash

nohup python -u train_sampling.py --task ppi_n >logs/test_lt/ppi/ppi_n 2>&1 &
#python -u train_sampling.py --task ppi_n