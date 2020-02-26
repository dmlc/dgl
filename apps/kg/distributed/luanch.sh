SCRIPT_PATH=~/dgl/apps/kg/distributed
SCRIPT_FILE=./freebase_transe_l2.sh
KEY=~/mctt.pem

ssh -i $KEY ubuntu@18.221.176.194 'cd '$SCRIPT_PATH'; '$SCRIPT_FILE' 8 16' &
ssh -i $KEY ubuntu@3.17.157.253 'cd '$SCRIPT_PATH'; '$SCRIPT_FILE' 16 24' &
ssh -i $KEY ubuntu@13.59.205.163 'cd '$SCRIPT_PATH'; '$SCRIPT_FILE' 24 32' &
$SCRIPT_FILE 0 8