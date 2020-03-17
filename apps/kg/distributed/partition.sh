#!/bin/bash

##################################################################################
# User runs this script to partition a graph using METIS
##################################################################################
DATA_PATH=$1
DATA_SET=$2
K=$3

# partition graph
python3 ../partition.py --dataset $DATA_SET -k $K --data_path $DATA_PATH

# copy related file to partition
PART_ID=0
while [ $PART_ID -lt $K ]
do
    cp $DATA_PATH/$DATA_SET/relation* $DATA_PATH/$DATA_SET/partition_$PART_ID
    let PART_ID+=1
done