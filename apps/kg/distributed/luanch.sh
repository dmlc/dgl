ssh -i mctt.pem ubuntu@18.221.176.194 'cd ~/dgl/apps/kg/distributed; ./run_server.sh & ./run_client.sh' &
ssh -i mctt.pem ubuntu@3.17.157.253 'cd ~/dgl/apps/kg/distributed; ./run_server.sh & ./run_client.sh' &
ssh -i mctt.pem ubuntu@13.59.205.163 'cd ~/dgl/apps/kg/distributed; ./run_server.sh & ./run_client.sh' &
./run_server.sh & 
./run_client.sh