ssh -i ~/mctt.pem ubuntu@18.221.176.194 'cd ~/dgl/apps/kg/distributed; ./server.sh & ./worker.sh' &
ssh -i ~/mctt.pem ubuntu@3.17.157.253 'cd ~/dgl/apps/kg/distributed; ./server.sh & ./worker.sh' &
ssh -i ~/mctt.pem ubuntu@13.59.205.163 'cd ~/dgl/apps/kg/distributed; ./server.sh & ./worker.sh' &
./server.sh & 
./worker.sh