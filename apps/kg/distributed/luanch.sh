SCRIPT_PATH=~/dgl/apps/kg/distributed
SCRIPT_FILE=./freebase_transe_l2.sh
KEY=~/mctt.pem

ssh -i $KEY ubuntu@13.58.56.18 'cd '$SCRIPT_PATH'; '$SCRIPT_FILE' 10 20' &
ssh -i $KEY ubuntu@18.189.143.5 'cd '$SCRIPT_PATH'; '$SCRIPT_FILE' 20 30' &
ssh -i $KEY ubuntu@18.191.82.244 'cd '$SCRIPT_PATH'; '$SCRIPT_FILE' 30 40' &
$SCRIPT_FILE 0 10