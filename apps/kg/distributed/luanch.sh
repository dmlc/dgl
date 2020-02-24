SCRIPT_PATH=~/dgl/apps/kg/distributed
SCRIPT_FILE=./freebase_transe_l2.sh
KEY=~/mctt.pem

ssh -i $KEY ubuntu@18.221.176.194 cd $SCRIPT_PATH; $SCRIPT_FILE 13 26 &
ssh -i $KEY ubuntu@3.17.157.253 cd $SCRIPT_PATH; $SCRIPT_FILE 26 39 &
ssh -i $KEY ubuntu@13.59.205.163 cd $SCRIPT_PATH; $SCRIPT_FILE 39 52 &
$SCRIPT_FILE 0 13