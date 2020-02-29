SCRIPT_PATH=~/dgl/apps/kg/distributed
SCRIPT_FILE=./fb15k_transe_l2.sh
KEY=~/mctt.pem

ssh -i $KEY ubuntu@18.216.60.5 'cd '$SCRIPT_PATH'; '$SCRIPT_FILE' 8 16' &
ssh -i $KEY ubuntu@3.136.27.4 'cd '$SCRIPT_PATH'; '$SCRIPT_FILE' 16 24' &
ssh -i $KEY ubuntu@52.14.94.4 'cd '$SCRIPT_PATH'; '$SCRIPT_FILE' 24 32' &
$SCRIPT_FILE 0 8