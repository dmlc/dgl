SCRIPT_PATH=~/dgl/apps/kg/distributed
SCRIPT_FILE=./freebase_distmult.sh
KEY=~/mctt.pem

ssh -i $KEY ubuntu@18.216.60.5 'cd '$SCRIPT_PATH'; '$SCRIPT_FILE' 13 26' &
ssh -i $KEY ubuntu@3.136.27.4 'cd '$SCRIPT_PATH'; '$SCRIPT_FILE' 26 39' &
ssh -i $KEY ubuntu@52.14.94.4 'cd '$SCRIPT_PATH'; '$SCRIPT_FILE' 39 52' &
$SCRIPT_FILE 0 13