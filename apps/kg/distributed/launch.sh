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
LINE_LOW=2
LINE_HIGH=$machine_count
while [ $LINE_LOW -lt $LINE_HIGH ]
do
    ip=$(awk 'NR=='$LINE_LOW' {print $1}' ip_config.txt)
    let LINE_LOW+=1
    echo yes | ssh -i $ssh_key $user_name@$ip 'cd '$script_path'; '$script_file' '$i' '$server_count' ' &
done

# run command on local machine
$script_file 1 $server_count