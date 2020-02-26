SCRIPT_PATH=~/dgl/apps/kg/distributed
SCRIPT_FILE=./freebase_transe_l2.sh
KEY=~/mctt.pem

ssh -i $KEY ubuntu@13.58.56.18 'cd '$SCRIPT_PATH'; '$SCRIPT_FILE' 8 16' &
ssh -i $KEY ubuntu@18.189.143.5 'cd '$SCRIPT_PATH'; '$SCRIPT_FILE' 16 24' &
ssh -i $KEY ubuntu@18.191.82.244 'cd '$SCRIPT_PATH'; '$SCRIPT_FILE' 24 32' &
$SCRIPT_FILE 0 8