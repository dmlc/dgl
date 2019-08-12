DGLBACKEND=mxnet python3 ./server.py --id 0 &
DGLBACKEND=mxnet python3 ./server.py --id 1 &
DGLBACKEND=mxnet python3 ./client.py --id 0 &
DGLBACKEND=mxnet python3 ./client.py --id 1