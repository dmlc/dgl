## Usage of DGL distributed KVStore

This is a simple example shows how to use DGL distributed KVStore on MXNet locally. In this example, we start 4 servers and 4 clients, and you can first run the command:

    ./run_server.sh

And when you see the message

    start server 1 on 127.0.0.1:50051
    start server 2 on 127.0.0.1:50052
    start server 0 on 127.0.0.1:50050
    start server 3 on 127.0.0.1:50053

you can start client by:

    ./run_client.sh
    


