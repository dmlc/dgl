#!/bin/sh

g++ norm.cpp -o norm

cd liblinear
make
cd ..

cd program
g++ -O2 preprocess.cpp -o preprocess
g++ -O2 score.cpp -o score
