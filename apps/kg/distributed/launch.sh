##################################################################################
# User runs this script to launch distrobited jobs on cluster
##################################################################################
script_path=~/dgl/apps/kg/distributed
script_file=./freebase_transe_l2.sh
user_name=ubuntu
ssh_key=~/mctt.pem

server_count=$(awk 'NR==1 {print $3}' ip_config.txt)

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
    ssh -i $ssh_key $user_name@$ip 'cd '$script_path'; '$script_file' '$s_id' '$server_count' ' &
done

# run command on local machine
$script_file 0 $server_count