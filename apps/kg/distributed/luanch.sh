SCRIPT_PATH=~/dgl/apps/kg/distributed
SCRIPT_FILE=./freebase_transe_l2.sh
KEY=~/mctt.pem

ssh -i $KEY ubuntu@13.58.56.18 'cd '$SCRIPT_PATH'; '$SCRIPT_FILE' 13 26' &
ssh -i $KEY ubuntu@18.189.143.5 'cd '$SCRIPT_PATH'; '$SCRIPT_FILE' 26 39' &
ssh -i $KEY ubuntu@18.191.82.244 'cd '$SCRIPT_PATH'; '$SCRIPT_FILE' 39 52' &
$SCRIPT_FILE 0 13