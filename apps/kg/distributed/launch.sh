##################################################################################
# User run this script to launch distrobited jobs on cluster
##################################################################################
script_path=~/dgl/apps/kg/distributed
script_file=./freebase_transe_l2.sh
user_name=ubuntu
ssh_key=~/mctt.pem

# Delete the temp file
rm *-shape

server_count=$(awk 'NR==1 {print $3}' ip_config.txt)
machine_count=$(awk 'END{print NR}' ip_config.txt)

# run command on remote machine
for i in {2..$machine_count}
do
    ip = $(awk a=$i 'NR==$a {print $1}' ip_config.txt)
    echo yes | ssh -i $ssh_key $user_nameu@ip 'cd '$script_path'; '$script_file' $i $server_count' &
done

# run command on local machine
$script_file 1 $server_count