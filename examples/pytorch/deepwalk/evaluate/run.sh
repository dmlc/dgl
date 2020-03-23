#!/bin/bash

vector_file=$1
times=10
portion=0.01
output_file=result.txt

rm -rf ${output_file}
touch ${output_file}

vocab_file=$2
label_file=$3
workspace=workspace/

./program/preprocess -vocab ${vocab_file} -vector ${vector_file} -label ${label_file} -output ${workspace} -debug 2 -binary 1 -times ${times} -portion ${portion}

for (( i = 0; i < ${times} ; i ++ ))
do
	./liblinear/train -s 0 -q ${workspace}train${i} ${workspace}model${i}
done

for (( i = 0; i < ${times} ; i ++ ))
do
	./liblinear/predict -b 1 -q ${workspace}test${i} ${workspace}model${i} ${workspace}predict${i}
done

for (( i = 0; i < ${times} ; i ++ ))
do
	./program/score -predict ${workspace}predict${i} -candidate ${workspace}can${i} >> ${output_file}
done
