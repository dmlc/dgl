#!/bin/bash

##################################################################################
# User runs this script to launch distrobited jobs on cluster
##################################################################################
script_path=$1
script_file=$2
user_name=$3
ssh_key=$4

server_count=$(awk 'NR==1 {print $3}' ip_config.txt)
machine_count=$(awk 'END{print NR}' ip_config.txt)

# run command on remote machine
LINE_LOW=2
LINE_HIGH=$(awk 'END{print NR}' ip_config.txt)
let LINE_HIGH+=1
s_id=0
while [ $LINE_LOW -lt $LINE_HIGH ]
do
    ip=$(awk 'NR=='$LINE_LOW' {print $1}' ip_config.txt)
    let LINE_LOW+=1
    let s_id+=1
    if test -z "$ssh_key" 
    then
        ssh $user_name@$ip 'cd '$script_path'; '$script_file' '$s_id' '$server_count' '$machine_count'' &
    else
        ssh -i $ssh_key $user_name@$ip 'cd '$script_path'; '$script_file' '$s_id' '$server_count' '$machine_count'' &
    fi
done

# run command on local machine
$script_file 0 $server_count $machine_count