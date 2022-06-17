#!/bin/bash


cur_dir=$(pwd)
host_count=`cat hostfile | wc -l`
graph_name="order"
perhost_part=2
current_host=`ifconfig | grep -Eo 'inet (addr:)?([0-9]*\.){3}[0-9]*' | grep -Eo '([0-9]*\.){3}[0-9]*' | grep -v '127.0.0.1'`

echo "metis creation start"

##Nodes 
`python3 metis_creation.py -n ${graph_name}`

echo "metis creation ends"

echo "directory creation starts"
while read p; do
  if [ "$p" != "$current_host" ]; then
    `ssh ${p} "mkdir -p ${cur_dir}" < /dev/null`
  fi
done <hostfile

echo "directory creation ends"

echo "partioning starts"
`mpirun --hostfile hostfile -np ${host_count} pm_dglpart ${graph_name} ${perhost_part} > mpirun.out`
echo "partioning ends"


echo "scp starts"
while read p; do
  if [ "$p" != "$current_host" ]; then
    `scp ${p}:${cur_dir}/* ./ < /dev/null`
  fi
done <hostfile
echo "scp ends"

echo "fetching removed edges starts"
`cat mpirun.out | grep "Duplicate edges with metadata" | awk -F'[][]' '{print $4}' > remove.csv`
echo "fetching removed edges ends"

echo "homo graph to herto graph starts"
`python3 substitute_to_hetero.py -n order -r remove.csv`
echo "homo graph to herto graph ends"
